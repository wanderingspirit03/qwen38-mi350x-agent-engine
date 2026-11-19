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
def _tile(pid, M, N, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr, XCD: tl.constexpr):
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    if XCD > 1:
        # workgroup i runs on XCD i % XCD: give each XCD a contiguous range of tiles (shared L2)
        g = npm * npn
        per = (g + XCD - 1) // XCD
        tall = g % XCD
        tall = XCD if tall == 0 else tall
        x = pid % XCD
        loc = pid // XCD
        if x < tall:
            pid = x * per + loc
        else:
            pid = tall * per + (x - tall) * (per - 1) + loc
    per_group = GROUP_M * npn
    gid = pid // per_group
    first = gid * GROUP_M
    gsz = min(npm - first, GROUP_M)
    pid_m = first + (pid % per_group) % gsz
    pid_n = (pid % per_group) // gsz
    return pid_m, pid_n


@triton.jit(do_not_specialize=["M"])
def _f8mm(a_ptr, sa_ptr, b_ptr, sb_ptr, c_ptr, M, N, K, stride_c,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, XCD: tl.constexpr,
          SILU: tl.constexpr, SB: tl.constexpr):
    """SILU: b rows come in blocks of SB (gate SB rows, up SB rows, ...); c (M, N/2) = bf16(silu(g) * u) with g, u
    rounded to bf16 first (as hip.silu_mul). With SB = 16 the matching g and u sit in the same thread's registers."""
    pid_m, pid_n = _tile(tl.program_id(0), M, N, BM, BN, GROUP_M, XCD)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(K > 0)
    om = (pid_m * BM + tl.arange(0, BM)) % M
    on = (pid_n * BN + tl.arange(0, BN)) % N
    ok = tl.arange(0, BK)
    a_ptrs = a_ptr + om[:, None] * K + ok[None, :]
    b_ptrs = b_ptr + on[None, :] * K + ok[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K // BK):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0)
    sb = tl.load(sb_ptr + rn, mask=rn < N, other=0.0)
    acc = acc * sa[:, None] * sb[None, :]
    if SILU:
        g, u = tl.split(tl.permute(tl.reshape(acc, (BM, BN // (2 * SB), 2, SB)), (0, 1, 3, 2)))
        g = tl.reshape(g, (BM, BN // 2)).to(tl.bfloat16).to(tl.float32)
        u = tl.reshape(u, (BM, BN // 2)).to(tl.bfloat16).to(tl.float32)
        s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        rc = pid_n * (BN // 2) + tl.arange(0, BN // 2)
        tl.store(c_ptr + rm[:, None] * stride_c + rc[None, :], (s * u).to(tl.bfloat16),
                 mask=(rm[:, None] < M) & (rc[None, :] < N // 2))
    else:
        tl.store(c_ptr + rm[:, None] * stride_c + rn[None, :], acc.to(tl.bfloat16),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


BIG = (256, 256, 128, 8, 8, 2)      # BM, BN, BK, GROUP_M, num_warps, num_stages
SMALL = (128, 128, 128, 8, 4, 2)


def f8mm(a, sa, b, sb, out, cfg=BIG, silu=False):
    """a (M, K) fp8, sa (M, 1) fp32, b (N, K) fp8 (interleave()d when silu), sb (1, N) fp32 -> out (M, N) bf16,
    or out (M, N/2) = silu(gate) * up when silu. Rows of out may be strided."""
    M, K = a.shape
    N = b.shape[0]
    BM, BN, BK, GM, nw, ns = cfg
    assert K % BK == 0 and out.stride(1) == 1
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    return _f8mm[grid](a, sa, b, sb, out, M, N, K, out.stride(0), BM, BN, BK, GM, 8, silu, SILU_BLK,
                       num_warps=nw, num_stages=ns)


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
            for tag, cfg in (("big", BIG), ("small", SMALL)):
                t = timeit(lambda: f8mm(xq, xs, wq, ws, c1, cfg))
                err = ((c1.float() - c0.float()).norm() / c0.float().norm()).item()
                line += f" | {tag} {t * 1e6:7.1f} us {fl / t / 1e15:4.2f} err {err:.0e}"
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

        def fused(cfg):
            f8mm(xq, xs, wqi, wsi, act, cfg, silu=True)
            H.quant_fp8(act, aq1, s1)

        tb = timeit(base)
        line = f"silu M={M}: hipblaslt+silu_mul_quant {tb * 1e6:.0f} us"
        for tag, cfg in (("big", BIG), ("small", SMALL)):
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
            for tag, cfg in (("big", BIG), ("small", SMALL)):
                t = timeit(lambda: f8mm(xq, xs, wq, ws, c, cfg), it=20, rep=2)
                line += f" {tag} {fl / t / 1e15:4.2f}"
            print(line, flush=True)


if __name__ == "__main__":
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if sys.argv[1] == "bench":
        bench([int(v) for v in arg.split(",")] if arg else [1024, 2048, 3500, 5600],
              sys.argv[3].split(",") if len(sys.argv) > 3 else list(SHAPES))
    elif sys.argv[1] == "silu":
        bench_silu([int(v) for v in arg.split(",")] if arg else [512, 768, 1024, 2048, 3500, 5600])
    elif sys.argv[1] == "power":
        bench_power()
    elif sys.argv[1] == "sweep":
        sweep(arg.split(",") if arg else list(SHAPES))
