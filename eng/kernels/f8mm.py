"""FP8 (e4m3fn) GEMMs for prefill in Triton, with rowwise scales and a fused SiLU*up epilogue.
c (M, N) = (a (M, K) fp8 * sa (M, 1)) @ (b (N, K) fp8 * sb (1, N))^T, fp32 accumulation, bf16 out.
Bench: python3 -m eng.kernels.f8mm bench [Ms] [shapes] | silu [Ms] | power | sweep [shapes]"""
import sys
import time

import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
SILU_BLK = 16       # gate_up rows interleaved in blocks of 16 (gate 16, up 16, ...): see interleave()


@triton.jit
def _xcd(pid, n, XCD: tl.constexpr):
    """Workgroup pid runs on XCD pid % XCD: map it so each XCD gets a contiguous range of 0..n-1 (shared L2)."""
    per = (n + XCD - 1) // XCD
    tall = n % XCD
    tall = XCD if tall == 0 else tall
    x = pid % XCD
    loc = pid // XCD
    if x < tall:
        r = x * per + loc
    else:
        r = tall * per + (x - tall) * (per - 1) + loc
    return r


@triton.jit
def _group(t, npm, npn, GROUP_M: tl.constexpr):
    per_group = GROUP_M * npn
    first = (t // per_group) * GROUP_M
    gsz = min(npm - first, GROUP_M)
    return first + (t % per_group) % gsz, (t % per_group) // gsz


@triton.jit
def _mm_tile(a_ptr, sa_ptr, b_ptr, sb_ptr, c_ptr, M, N, K, stride_c, off_m, off_n,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SILU: tl.constexpr, SB: tl.constexpr,
             ABLK: tl.constexpr = False):
    om = (off_m + tl.arange(0, BM)) % M
    on = (off_n + tl.arange(0, BN)) % N
    ok = tl.arange(0, BK)
    a_ptrs = a_ptr + om[:, None] * K + ok[None, :]
    b_ptrs = b_ptr + on[None, :] * K + ok[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if ABLK:
        # A scales per (row, BK block of K): sa (M, K / BK)
        s_ptrs = sa_ptr + om * (K // BK)
        for _ in range(0, K // BK):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            sk = tl.load(s_ptrs)
            acc += tl.dot(a, b) * sk[:, None]
            a_ptrs += BK
            b_ptrs += BK
            s_ptrs += 1
    else:
        for _ in range(0, K // BK):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK
    rm = off_m + tl.arange(0, BM)
    rn = off_n + tl.arange(0, BN)
    if ABLK:
        sa = tl.full((BM,), 1.0, tl.float32)
    else:
        sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0)
    sb = tl.load(sb_ptr + rn, mask=rn < N, other=0.0)
    acc = acc * sa[:, None] * sb[None, :]
    if SILU:
        g, u = tl.split(tl.permute(tl.reshape(acc, (BM, BN // (2 * SB), 2, SB)), (0, 1, 3, 2)))
        g = tl.reshape(g, (BM, BN // 2)).to(tl.bfloat16).to(tl.float32)
        u = tl.reshape(u, (BM, BN // 2)).to(tl.bfloat16).to(tl.float32)
        s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        rc = off_n // 2 + tl.arange(0, BN // 2)
        tl.store(c_ptr + rm[:, None] * stride_c + rc[None, :], (s * u).to(tl.bfloat16),
                 mask=(rm[:, None] < M) & (rc[None, :] < N // 2))
    else:
        tl.store(c_ptr + rm[:, None] * stride_c + rn[None, :], acc.to(tl.bfloat16),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit(do_not_specialize=["M", "base"])
def _f8mm(a_ptr, sa_ptr, b_ptr, sb_ptr, c_ptr, M, N, K, stride_c, base,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, XCD: tl.constexpr,
          TM: tl.constexpr, TN: tl.constexpr, SILU: tl.constexpr, SB: tl.constexpr, ABLK: tl.constexpr = False):
    """Big tiles (BM, BN) base, base+1, ... (in grouped order), each split into TM x TN sub-tiles (BM/TM, BN/TN),
    one per program. SILU: b rows come in blocks of SB (gate SB rows, up SB rows, ...); c (M, N/2) =
    bf16(silu(g) * u) with g, u rounded to bf16 first (as hip.silu_mul). With SB = 16 the matching g and u sit in
    the same thread's registers."""
    tl.assume(K > 0)
    tl.assume(M > 0)
    q = _xcd(tl.program_id(0), tl.num_programs(0), XCD)
    pid_m, pid_n = _group(base + q // (TM * TN), tl.cdiv(M, BM), tl.cdiv(N, BN), GROUP_M)
    s = q % (TM * TN)
    _mm_tile(a_ptr, sa_ptr, b_ptr, sb_ptr, c_ptr, M, N, K, stride_c, pid_m * BM + (s // TN) * (BM // TM),
             pid_n * BN + (s % TN) * (BN // TN), BM // TM, BN // TN, BK, SILU, SB, ABLK)


BIG = (256, 256, 128, 8, 8, 2)      # BM, BN, BK, GROUP_M, num_warps, num_stages
NCU = torch.cuda.get_device_properties(0).multi_processor_count
SMALL_EFF = 0.6                     # throughput of a quarter tile relative to a big one (fits the split/no-split bench)


def plan(M, N, cfg=BIG, split=(2, 2)):
    """(n_main, n_big): big tiles fill whole waves (0..n_main-1); the rest (n_main..n_big-1) are split into small
    ones when that is quicker."""
    BM, BN = cfg[:2]
    n_big = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    left = n_big % NCU
    sub = split[0] * split[1]
    if left == 0 or sub == 1 or triton.cdiv(left * sub, NCU) / sub / SMALL_EFF >= 1:
        return n_big, n_big
    return n_big - left, n_big


def f8mm(a, sa, b, sb, out, cfg=BIG, silu=False, split=(2, 2), ablk=False):
    """a (M, K) fp8, sa (M, 1) fp32, b (N, K) fp8 (interleave()d when silu), sb (1, N) fp32 -> out (M, N) bf16,
    or out (M, N/2) = silu(gate) * up when silu. Rows of out may be strided."""
    M, K = a.shape
    N = b.shape[0]
    BM, BN, BK, GM, nw, ns = cfg
    assert K % BK == 0 and out.stride(1) == 1
    n_main, n_big = plan(M, N, cfg, split)
    if n_main:
        _f8mm[(n_main,)](a, sa, b, sb, out, M, N, K, out.stride(0), 0, BM, BN, BK, GM, 8, 1, 1, silu, SILU_BLK, ablk,
                         num_warps=nw, num_stages=ns)
    if n_big > n_main:
        TM, TN = split
        _f8mm[((n_big - n_main) * TM * TN,)](a, sa, b, sb, out, M, N, K, out.stride(0), n_main, BM, BN, BK, GM, 8,
                                             TM, TN, silu, SILU_BLK, ablk, num_warps=nw // max(1, (TM * TN) // 2),
                                             num_stages=ns)


def interleave(w, blk=SILU_BLK):
    """(2I, ...) [gate; up] -> rows in blocks of blk: (gate 0..blk-1, up 0..blk-1, gate blk.., up blk.., ...)."""
    I = w.shape[0] // 2
    return torch.stack((w[:I].reshape(I // blk, blk, -1), w[I:].reshape(I // blk, blk, -1)), 1).reshape(w.shape)


# ---------------------------------------------------------------------------------------------- benchmarks
SHAPES = {"gate_up": (34816, 5120), "down": (5120, 17408), "in_proj": (16480, 5120), "out": (5120, 6144),
          "qkv": (14336, 5120)}


def q8(x):
    s = x.float().abs().amax(1, keepdim=True).clamp(min=1e-8) / 448.0
    return (x.float() / s).to(FP8), s


def timeit(f, it=50, rep=3):
    best = 1e9
    for _ in range(rep):
        f()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(it):
            f()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / 1e3 / it)
    return best


def weights(name):
    N, K = SHAPES[name]
    wq, ws = q8(torch.randn(N, K, device="cuda") * 0.02)
    return wq, ws.reshape(1, -1).contiguous()


def hip_mm(xq, xs, wq, ws, c):
    torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16, out=c)


def bench(Ms, names):
    torch.manual_seed(0)
    for name in names:
        wq, ws = weights(name)
        N, K = wq.shape
        for M in Ms:
            xq, xs = q8(torch.randn(M, K, device="cuda"))
            c0 = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            c1 = torch.empty_like(c0)
            fl = 2 * M * N * K
            t0 = timeit(lambda: hip_mm(xq, xs, wq, ws, c0))
            line = f"{name:8s} M={M:5d} hipblaslt {t0 * 1e6:7.1f} us {fl / t0 / 1e15:4.2f} PF/s"
            for sp in ((1, 1), (2, 2), (1, 2), (2, 1)):
                c1.zero_()
                t = timeit(lambda: f8mm(xq, xs, wq, ws, c1, split=sp))
                err = ((c1.float() - c0.float()).norm() / c0.float().norm()).item()
                nm, nb = plan(M, N, BIG, sp)
                line += f" | {sp[0]}x{sp[1]} {nb - nm}: {t * 1e6:6.1f} {fl / t / 1e15:4.2f}" + (f" ERR {err:.0e}" if err > 1e-3 else "")
            print(line, flush=True)


def bench_silu(Ms):
    from .. import hip as H
    torch.manual_seed(0)
    wq, ws = weights("gate_up")
    N, K = wq.shape
    I = N // 2
    wqi = interleave(wq.view(torch.uint8)).view(FP8)
    wsi = interleave(ws.reshape(-1, 1)).reshape(1, -1).contiguous()
    for M in Ms:
        xq, xs = q8(torch.randn(M, K, device="cuda"))
        gu = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        act = torch.empty(M, I, device="cuda", dtype=torch.bfloat16)
        aq0, aq1 = (torch.empty(M, I, device="cuda", dtype=FP8) for _ in range(2))
        s0, s1 = (torch.empty(M, 1, device="cuda") for _ in range(2))

        def base():
            hip_mm(xq, xs, wq, ws, gu)
            H.silu_mul_quant(gu, aq0, s0)

        def fused(sp):
            f8mm(xq, xs, wqi, wsi, act, silu=True, split=sp)
            H.quant_fp8(act, aq1, s1)

        tb = timeit(base)
        line = f"silu M={M}: hipblaslt+silu_mul_quant {tb * 1e6:.0f} us"
        for tag, cfg in (("split", (2, 2)), ("nosplit", (1, 1))):
            t = timeit(lambda: fused(cfg))
            fused(cfg)
            base()
            same = (aq0.view(torch.uint8) == aq1.view(torch.uint8)).float().mean().item()
            line += f" | fused {tag} + quant_fp8 {t * 1e6:.0f} us (same fp8 bytes {same:.4f})"
        print(line, flush=True)


def power_sample(stop, out):
    import json
    import subprocess
    while not stop.is_set():
        r = subprocess.run(["amd-smi", "metric", "-p", "-c", "--json"], capture_output=True, text=True)
        try:
            out.append(json.loads(r.stdout)["gpu_data"][0])
        except Exception:
            pass


def sustained(f, flops, secs=3.0):
    """Average PF/s over secs of back-to-back calls, with the mean socket power and gfx clock of amd-smi samples."""
    import threading
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    smp, stop = [], threading.Event()
    th = threading.Thread(target=power_sample, args=(stop, smp))
    th.start()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < secs:
        for _ in range(20):
            f()
        n += 20
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    stop.set()
    th.join()
    pw = [s["power"]["socket_power"]["value"] for s in smp]
    clk = [s["clock"]["gfx_0"]["clk"]["value"] for s in smp]
    return flops * n / dt / 1e15, sum(pw) / max(len(pw), 1), sum(clk) / max(len(clk), 1)


def bench_power():
    torch.manual_seed(0)
    for name, M in (("gate_up", 3500), ("gate_up", 5600), ("in_proj", 3500), ("down", 5600)):
        wq, ws = weights(name)
        N, K = wq.shape
        xq, xs = q8(torch.randn(M, K, device="cuda"))
        c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        fl = 2 * M * N * K
        for tag, f in (("hipblaslt", lambda: hip_mm(xq, xs, wq, ws, c)), ("triton", lambda: f8mm(xq, xs, wq, ws, c))):
            pf, pw, clk = sustained(f, fl)
            print(f"{name} M={M} {tag}: {pf:.2f} PF/s sustained, {pw:.0f} W, gfx {clk:.0f} MHz, "
                  f"{pw / pf:.0f} pJ/kFLOP", flush=True)


def sweep(names):
    torch.manual_seed(0)
    for name in names:
        wq, ws = weights(name)
        N, K = wq.shape
        for M in range(512, 8193, 512):
            xq, xs = q8(torch.randn(M, K, device="cuda"))
            c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            fl = 2 * M * N * K
            t0 = timeit(lambda: hip_mm(xq, xs, wq, ws, c), it=20, rep=2)
            line = f"sweep {name:8s} M={M:5d} hip {fl / t0 / 1e15:4.2f}"
            for tag, sp in (("nosplit", (1, 1)), ("split", (2, 2))):
                t = timeit(lambda: f8mm(xq, xs, wq, ws, c, split=sp), it=20, rep=2)
                line += f" {tag} {fl / t / 1e15:4.2f}"
            print(line, flush=True)


def bench_ablk(Ms):
    """down GEMM with per-(row, 128) activation scales vs rowwise (Triton) vs hipBLASLt."""
    torch.manual_seed(0)
    wq, ws = weights("down")
    N, K = wq.shape
    for M in Ms:
        x = torch.randn(M, K, device="cuda")
        xq, xs = q8(x)
        xb = x.view(M, K // 128, 128)
        bs = xb.abs().amax(2).clamp(min=1e-8) / 448.0
        xbq = (xb / bs[..., None]).to(FP8).view(M, K)
        bs = bs.contiguous()
        c0, c1, c2 = (torch.empty(M, N, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        fl = 2 * M * N * K
        t0 = timeit(lambda: hip_mm(xq, xs, wq, ws, c0))
        t1 = timeit(lambda: f8mm(xq, xs, wq, ws, c1))
        t2 = timeit(lambda: f8mm(xbq, bs, wq, ws, c2, ablk=True))
        ref = x @ (wq.float() * ws.reshape(-1, 1)).t()
        e = lambda c: ((c.float() - ref).norm() / ref.norm()).item()
        print(f"ablk down M={M}: hip {t0 * 1e6:.0f} us | rowwise {t1 * 1e6:.0f} | block {t2 * 1e6:.0f} us "
              f"{fl / t2 / 1e15:.2f} PF/s | err vs exact-x: row {e(c1):.2e} block {e(c2):.2e}", flush=True)


def table(names, Ms):
    torch.manual_seed(0)
    for name in names:
        wq, ws = weights(name)
        N, K = wq.shape
        res = []
        for M in Ms:
            xq, xs = q8(torch.randn(M, K, device="cuda"))
            c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            t0 = timeit(lambda: hip_mm(xq, xs, wq, ws, c), it=20, rep=3)
            t1 = timeit(lambda: f8mm(xq, xs, wq, ws, c), it=20, rep=3)
            res.append(f"{M}:{t0 / t1:.2f}")
        print(f"table {name} (hip time / triton time):", " ".join(res), flush=True)


if __name__ == "__main__":
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if sys.argv[1] == "bench":
        bench([int(v) for v in arg.split(",")] if arg else [1024, 2048, 3500, 5600],
              sys.argv[3].split(",") if len(sys.argv) > 3 else list(SHAPES))
    elif sys.argv[1] == "silu":
        bench_silu([int(v) for v in arg.split(",")] if arg else [512, 768, 1024, 2048, 3500, 5600])
    elif sys.argv[1] == "power":
        bench_power()
    elif sys.argv[1] == "ablk":
        bench_ablk([int(v) for v in arg.split(",")] if arg else [1000, 2048, 3500, 5600])
    elif sys.argv[1] == "table":
        table(["down", "in_proj", "out", "qkv"], list(range(256, 8193, 256)))
    elif sys.argv[1] == "sweep":
        sweep(arg.split(",") if arg else list(SHAPES))
