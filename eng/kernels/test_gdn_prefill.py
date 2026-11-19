"""Test + benchmark for eng.kernels.gdn_prefill.
  python3 -m eng.kernels.test_gdn_prefill [--quick] [--prof]"""
import sys
import torch
import torch.nn.functional as F
from eng import ref_ops as R
from eng import hip as H
from eng.kernels.gdn_prefill import gdn_prefill, HK, HV, D

NCH = 10240
VD = 6144
PROJ = NCH + VD + 2 * HV


def engine_gdn(proj, hist, cw, aneg, dtb, gw, S, out, bufs={}):
    """The engine's GDN prefill: gdn_conv, then gdn_prefill with the gated norm."""
    T = proj.shape[0]
    if T not in bufs:
        bufs[T] = (torch.empty(T, NCH, device="cuda", dtype=torch.bfloat16),
                   torch.empty(T, 2, HV, device="cuda", dtype=torch.float32))
    y, gb = bufs[T]
    H.gdn_conv(proj, hist, cw, aneg, dtb, y, gb)
    return gdn_prefill(y[:, :2048].view(T, HK, D), y[:, 2048:4096].view(T, HK, D), y[:, 4096:].view(T, HV, D),
                       gb[:, 0], gb[:, 1], S, out=out, z=proj[:, NCH:NCH + VD], gw=gw)


def make_inputs(T, seed=0, dev="cuda"):
    gen = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, device=dev, generator=gen)
    q = rn(T, HK, D).bfloat16()
    k = rn(T, HK, D).bfloat16()
    v = rn(T, HV, D).bfloat16()
    g = -F.softplus(rn(T, HV)) * torch.exp(torch.rand(T, HV, device=dev, generator=gen) * 4 - 3)
    beta = torch.sigmoid(rn(T, HV))
    S0 = rn(HV, D, D) * 0.5
    return q, k, v, g, beta, S0


def reference(q, k, v, g, beta, S0, chunked=False):
    qf = (R.l2norm(q) * D ** -0.5).repeat_interleave(3, 1)
    kf = R.l2norm(k).repeat_interleave(3, 1)
    S = S0.clone()
    if chunked:
        o = R.gdn_chunked(qf, kf, v.float(), g, beta, S)
    else:
        o = R.gdn_recurrent(qf, kf, v.float(), g, beta, S)
    return o, S


def err(a, b):
    e = (a.float() - b.float()).abs().max().item()
    return f"{e:.2e} ({e / b.float().abs().max().item():.1e})"


def check(Ts=(1, 37, 64, 1000, 4096)):
    print("gdn_prefill vs gdn_recurrent (fp32); errors: max abs (max abs / max |ref|)")
    for T in Ts:
        q, k, v, g, beta, S0 = make_inputs(T, seed=T)
        o_ref, S_ref = reference(q, k, v, g, beta, S0)
        S = S0.transpose(1, 2).contiguous()
        o = gdn_prefill(q, k, v, g, beta, S).float()
        S = S.transpose(1, 2)
        rms_o = ((o - o_ref).pow(2).mean() / o_ref.pow(2).mean()).sqrt().item()
        eb = (o_ref.bfloat16().float() - o_ref).abs().max().item()
        print(f"T={T:5d}  o: {err(o, o_ref)} rel-rms {rms_o:.1e}   state: {err(S, S_ref)}   [bf16 rounding of o: {eb:.1e}]")
        S2 = S0.clone()
        o2 = gdn_prefill(q, k, v, g, beta, S2, state_kv=True).float()
        assert (o2 - o).abs().max().item() == 0 and (S2 - S).abs().max().item() == 0


def make_proj(T, seed=0, dev="cuda"):
    gen = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, device=dev, generator=gen)
    proj = torch.empty(T, PROJ + 32, device=dev, dtype=torch.bfloat16)[:, :PROJ]     # padded row stride
    proj.copy_(rn(T, PROJ))
    proj[:, NCH + VD + HV:] = rn(T, HV) - 2
    hist = rn(3, NCH).bfloat16()
    cw = (rn(NCH, 4) * 0.4).bfloat16()
    aneg = -torch.exp(rn(HV) * 0.5)
    dtb = rn(HV) * 0.5
    gw = (1 + 0.1 * rn(D)).bfloat16()
    S0 = rn(HV, D, D) * 0.5
    return proj, hist, cw, aneg, dtb, gw, S0


def reference_proj(proj, hist, cw, aneg, dtb, gw, S0):
    T = proj.shape[0]
    qkv = proj[:, :NCH]
    xx = torch.cat([hist, qkv], 0).float()
    y = F.silu(sum(xx[j:j + T] * cw[:, j].float() for j in range(4))).bfloat16()
    q, k, v = y.split([2048, 2048, VD], -1)
    b = proj[:, NCH + VD:NCH + VD + HV].float()
    a = proj[:, NCH + VD + HV:].float()
    g = aneg * F.softplus(a + dtb)
    beta = torch.sigmoid(b)
    o, S = reference(q.reshape(T, HK, D), k.reshape(T, HK, D), v.reshape(T, HV, D), g, beta, S0, chunked=True)
    z = proj[:, NCH:NCH + VD].reshape(T, HV, D)
    return R.rmsnorm_gated(o.bfloat16(), z, gw).reshape(T, VD), S


def check_proj(Ts=(1, 37, 1000, 4096)):
    print("gdn_conv + gdn_prefill + gated norm vs reference (gdn_chunked fp32)")
    for T in Ts:
        proj, hist, cw, aneg, dtb, gw, S0 = make_proj(T, seed=T)
        o_ref, S_ref = reference_proj(proj, hist, cw, aneg, dtb, gw, S0)
        S = S0.transpose(1, 2).contiguous()
        out = torch.empty(T, VD, device="cuda", dtype=torch.bfloat16)
        engine_gdn(proj, hist, cw, aneg, dtb, gw, S, out)
        S = S.transpose(1, 2)
        rms = ((out.float() - o_ref.float()).pow(2).mean() / o_ref.float().pow(2).mean()).sqrt().item()
        print(f"T={T:5d}  out: {err(out, o_ref)} rel-rms {rms:.1e}   state: {err(S, S_ref)}")


def timeit(fn, iters=30, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def bench(Ts=(64, 420, 1000, 1500, 4096, 5600, 12000)):
    print("ms per call (all kernels and allocations): gdn_prefill (raw q k v g beta -> o) | engine_gdn (proj -> gated out)")
    for T in Ts:
        q, k, v, g, beta, S0 = make_inputs(T)
        S = S0.clone()
        t1 = timeit(lambda: gdn_prefill(q, k, v, g, beta, S))
        proj, hist, cw, aneg, dtb, gw, S0 = make_proj(T)
        out = torch.empty(T, VD, device="cuda", dtype=torch.bfloat16)
        t2 = timeit(lambda: engine_gdn(proj, hist, cw, aneg, dtb, gw, S, out))
        print(f"T={T:5d}  {t1:.4f} ms | {t2:.4f} ms")


def kernel_times(T):
    from torch.profiler import profile, ProfilerActivity
    proj, hist, cw, aneg, dtb, gw, S = make_proj(T)
    out = torch.empty(T, VD, device="cuda", dtype=torch.bfloat16)
    f = lambda: engine_gdn(proj, hist, cw, aneg, dtb, gw, S, out)
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            f()
        torch.cuda.synchronize()
    ev = [(e.key[:20], e.device_time_total / e.count) for e in prof.key_averages() if e.device_time_total > 0]
    print(f"T={T} kernels (us):", "  ".join(f"{k} {v:.1f}" for k, v in ev), f"  sum {sum(v for _, v in ev):.1f}")


def wy_compare(Ts=(1, 37, 64, 1000, 5600)):
    """HIP gdn_wy vs the Triton _wy_kernel: the same outputs, errors relative to max |ref|; timings."""
    import triton
    from eng.kernels import gdn_prefill as GP
    for T in Ts:
        q, k, v, g, beta, _ = make_inputs(T, seed=T)
        NC = triton.cdiv(T, 64)
        NC += NC % 2
        Tp = NC * 64
        bf = torch.bfloat16
        outs = []
        for hip in (False, True):
            o = dict(w=torch.zeros(Tp, HV, D, device="cuda", dtype=bf), uT=torch.zeros(NC, HV, D, 64, device="cuda", dtype=bf),
                     kdT=torch.zeros(NC, HV, D, 64, device="cuda", dtype=bf), G=torch.zeros(HV, Tp, device="cuda"),
                     qn=torch.zeros(Tp, HK, D, device="cuda", dtype=bf), M=torch.zeros(NC, HV, 64, 64, device="cuda", dtype=bf))
            f = lambda: GP.wy(q, k, v, g, beta, o["w"], o["uT"], o["kdT"], o["G"], o["qn"], o["M"], T, Tp, NC, hip=hip)
            f()
            outs.append((o, timeit(f) * 1e3))
        (a, ta), (b, tb) = outs
        errs = " ".join(f"{n} {((b[n].float() - a[n].float()).abs().max() / a[n].float().abs().max().clamp(min=1e-30)).item():.1e}"
                        for n in a)
        print(f"wy T={T:5d}: triton {ta:.1f} us, hip {tb:.1f} us; rel max err {errs}", flush=True)


def ckpt_check(T=4133, every=1024):
    """States stored by ckpt equal the final states of calls on the prefixes."""
    q, k, v, g, beta, S0 = make_inputs(T, seed=5)
    n = T // every
    ck = torch.full((n, *S0.shape), float("nan"), device="cuda")
    S = S0.clone()
    o = gdn_prefill(q, k, v, g, beta, S, ckpt=(every, ck))
    S_plain = S0.clone()
    o_plain = gdn_prefill(q, k, v, g, beta, S_plain)
    assert torch.equal(o, o_plain) and torch.equal(S, S_plain)
    for i in range(n):
        m = (i + 1) * every
        Si = S0.clone()
        gdn_prefill(q[:m], k[:m], v[:m], g[:m], beta[:m], Si)
        print(f"ckpt {i} (after {m} tokens): max abs diff {(ck[i] - Si).abs().max().item():.2e}")


if __name__ == "__main__":
    if "--ckpt" in sys.argv:
        ckpt_check()
        sys.exit()
    if "--wy" in sys.argv:
        wy_compare()
        sys.exit()
    if "--quick" not in sys.argv:
        check()
        check_proj()
    bench()
    if "--nw" in sys.argv:
        import eng.kernels.gdn_prefill as M
        for nw in (4, 8, 2):
            M.SCAN_NW = nw
            print("SCAN_NW", nw)
            check_proj((37, 1000))
            for T in (420, 1000, 4096):
                kernel_times(T)
        M.SCAN_NW = 2
    if "--prof" in sys.argv:
        for T in (420, 1000, 4096, 5600):
            kernel_times(T)
