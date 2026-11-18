"""Test + benchmark for eng.kernels.gdn_prefill.  python3 -m eng.kernels.test_gdn_prefill [--sweep]"""
import sys
import torch
import torch.nn.functional as F
from eng import ref_ops as R
from eng.kernels.gdn_prefill import gdn_prefill, HK, HV, D


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


def check(Ts=(1, 37, 64, 1000, 4096), **kw):
    print("correctness vs gdn_recurrent (fp32); errors: max abs (max abs / max |ref|)")
    for T in Ts:
        q, k, v, g, beta, S0 = make_inputs(T, seed=T)
        o_ref, S_ref = reference(q, k, v, g, beta, S0)
        S = S0.transpose(1, 2).contiguous()
        o = gdn_prefill(q, k, v, g, beta, S, **kw).float()
        S = S.transpose(1, 2)
        eo = (o - o_ref).abs().max().item()
        es = (S - S_ref).abs().max().item()
        mo = o_ref.abs().max().item()
        ms = S_ref.abs().max().item()
        rms_o = ((o - o_ref).pow(2).mean() / o_ref.pow(2).mean()).sqrt().item()
        # bf16 rounding of the exact output, for scale
        eb = (o_ref.bfloat16().float() - o_ref).abs().max().item()
        oc, Sc = reference(q, k, v, g, beta, S0, chunked=True)
        ec = (oc - o_ref).abs().max().item()
        print(f"T={T:5d}  o: {eo:.2e} ({eo / mo:.1e}) rel-rms {rms_o:.1e}   state: {es:.2e} ({es / ms:.1e})"
              f"   [bf16 rounding of o: {eb:.1e}; gdn_chunked fp32: {ec:.1e}]")


def timeit(fn, iters=50, warm=5):
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


def bench(Ts=(1, 64, 1000, 4096, 12000), **kw):
    print("timing (ms per call, whole gdn_prefill including buffer allocation and both kernels)", kw)
    for T in Ts:
        q, k, v, g, beta, S0 = make_inputs(T)
        S = S0.clone()
        t = timeit(lambda: gdn_prefill(q, k, v, g, beta, S, **kw))
        print(f"T={T:5d}  {t:.4f} ms   ({T / t / 1e3:.1f} Mtok/s per layer)")


def kernel_times(T=4096, **kw):
    from torch.profiler import profile, ProfilerActivity
    q, k, v, g, beta, S0 = make_inputs(T)
    S = S0.clone()
    for _ in range(3):
        gdn_prefill(q, k, v, g, beta, S, **kw)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            gdn_prefill(q, k, v, g, beta, S, **kw)
        torch.cuda.synchronize()
    print(f"T={T} kernels {kw}:")
    for e in prof.key_averages():
        if e.device_time_total > 0:
            print(f"   {e.key[:60]:60s} {e.device_time_total / e.count / 1e3:.4f} ms x{e.count // 10}")


if __name__ == "__main__":
    if "--quick" not in sys.argv:
        check()
        bench()
    sweep = [dict(), dict(BV=32, scan_warps=4, out_warps=4), dict(s_split=2)]
    for kw in (sweep if "--sweep" in sys.argv else sweep[:1]):
        kernel_times(4096, **kw)
