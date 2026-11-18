"""Test + benchmark of eng.kernels.attention against an fp32 PyTorch reference and torch SDPA.
Run: ./try tests/in_container.sh python3 -m eng.kernels.test_attention [quick]"""
import sys
import time

import torch
import torch.nn.functional as F

from .attention import attn_prefill, attn_decode, HEADS, KV_HEADS, HEAD_DIM, GROUP, SCALE

dev = "cuda"
LMAX = 16384


def ref_attn(q, k_cache, v_cache, n):
    """fp32 reference: q (T,24,256) at positions n-T..n-1; returns (T,24,256) fp32."""
    T = q.shape[0]
    out = torch.empty(T, HEADS, HEAD_DIM, dtype=torch.float32, device=q.device)
    pos = torch.arange(n - T, n, device=q.device)[:, None]
    mask = torch.arange(n, device=q.device)[None] <= pos                     # (T, n)
    for g in range(KV_HEADS):
        K = k_cache[:n, g].float()
        V = v_cache[:n, g].float()
        for hh in range(GROUP):
            h = g * GROUP + hh
            s = (q[:, h].float() @ K.T) * SCALE
            s = s.masked_fill(~mask, float("-inf"))
            out[:, h] = torch.softmax(s, -1) @ V
    return out


def make_cache(n, scale_k=1.0):
    k = torch.full((LMAX, KV_HEADS, HEAD_DIM), float("nan"), dtype=torch.bfloat16, device=dev)
    v = torch.full_like(k, float("nan"))
    k[:n] = (torch.randn(n, KV_HEADS, HEAD_DIM, device=dev) * scale_k).bfloat16()
    v[:n] = torch.randn(n, KV_HEADS, HEAD_DIM, device=dev).bfloat16()
    return k, v


def make_q(T, scale=1.0):
    # strided like the model's q slice of the (T, 24, 512) q+gate projection
    qg = (torch.randn(T, HEADS, 2 * HEAD_DIM, device=dev) * scale).bfloat16()
    return qg[..., :HEAD_DIM], qg[..., HEAD_DIM:]


def bench(fn, iters=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3          # us


def bench_graph(fn, iters=50):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(3):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / (3 * iters) * 1e3


def err(o, ref):
    d = (o.float() - ref).abs()
    return d.max().item(), d.mean().item()


def check(name, o, ref, tol=2e-2):
    mx, mean = err(o, ref)
    ok = mx < tol and torch.isfinite(o).all().item()
    print(f"  {'OK ' if ok else 'BAD'} {name}: max abs err {mx:.2e} mean {mean:.2e}", flush=True)
    return ok


def prefill_flops(T, p0):
    n_keys = T * p0 + T * (T + 1) / 2
    return 4 * HEADS * HEAD_DIM * n_keys


def sdpa_baselines(q, k, v, p0):
    T = q.shape[0]
    n = p0 + T
    from torch.nn.attention.bias import causal_lower_right
    qh = q.transpose(0, 1)[None]
    Kh = k[:n].transpose(0, 1)[None]
    Vh = v[:n].transpose(0, 1)[None]
    res = {}
    cases = {
        "sdpa gqa+lower_right": lambda: F.scaled_dot_product_attention(
            qh, Kh, Vh, attn_mask=causal_lower_right(T, n), enable_gqa=True),
        "sdpa expand+lower_right": lambda: F.scaled_dot_product_attention(
            qh, Kh.repeat_interleave(GROUP, 1), Vh.repeat_interleave(GROUP, 1), attn_mask=causal_lower_right(T, n)),
        "sdpa expand+bool mask": lambda: F.scaled_dot_product_attention(
            qh, Kh.repeat_interleave(GROUP, 1), Vh.repeat_interleave(GROUP, 1),
            attn_mask=torch.arange(n, device=dev)[None] <= torch.arange(p0, n, device=dev)[:, None]),
    }
    for name, fn in cases.items():
        try:
            res[name] = (bench(fn, iters=5, warm=2), fn()[0].transpose(0, 1))
        except Exception as e:
            print(f"  {name}: failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
    return res


def main():
    quick = "quick" in sys.argv
    torch.manual_seed(0)
    ok = True
    print("== correctness: prefill", flush=True)
    for T, p0, sc in [(1, 0, 1.0), (37, 0, 1.0), (300, 129, 1.0), (1000, 5000, 3.0), (2048, 0, 1.0), (1500, 700, 2.0)]:
        k, v = make_cache(p0 + T, sc)
        q, gate = make_q(T, sc)
        ref = ref_attn(q, k, v, p0 + T)
        ok &= check(f"prefill T={T} p0={p0} qk-scale={sc}", attn_prefill(q, k, v, p0), ref)
        if T == 300:
            ok &= check("prefill gated", attn_prefill(q, k, v, p0, gate=gate), ref * torch.sigmoid(gate.float()))
    print("== correctness: decode", flush=True)
    for T, n, sc in [(1, 1, 1.0), (1, 17, 1.0), (4, 4, 1.0), (8, 100, 1.0), (16, 3000, 1.0), (8, 12000, 3.0),
                     (16, 16000, 1.0), (3, 65, 1.0), (12, 777, 2.0), (2, 5000, 1.0)]:
        k, v = make_cache(n, sc)
        q, gate = make_q(T, sc)
        ref = ref_attn(q, k, v, n)
        nt = torch.tensor([n], dtype=torch.int32, device=dev)
        ok &= check(f"decode T={T} n={n} qk-scale={sc}", attn_decode(q, k, v, nt), ref)
        if T == 8:
            ok &= check("decode gated", attn_decode(q, k, v, nt, gate=gate), ref * torch.sigmoid(gate.float()))
    # graph: capture once with one n, replay with other n values
    k, v = make_cache(16000)
    q, _ = make_q(8)
    nt = torch.tensor([100], dtype=torch.int32, device=dev)
    out = torch.empty(8, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    attn_decode(q, k, v, nt, out=out)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        attn_decode(q, k, v, nt, out=out)
    for n in [100, 5000, 16000, 9]:
        nt.fill_(n)
        gr.replay()
        ok &= check(f"decode in graph, replay n={n}", out, ref_attn(q, k, v, n))
    print("ALL OK" if ok else "SOME FAILED", flush=True)

    print("== prefill timing (per layer)", flush=True)
    shapes = [(4096, 6000), (12000, 0), (512, 0), (512, 8000), (2048, 10000), (8192, 4000)]
    if quick:
        shapes = shapes[:1]
    for T, p0 in shapes:
        k, v = make_cache(p0 + T)
        q, _ = make_q(T)
        tf = prefill_flops(T, p0)
        print(f" T={T} p0={p0}  ({tf / 1e9:.0f} GFLOP)", flush=True)
        if T * (T + p0) <= 4096 * 10096:
            ref = ref_attn(q, k, v, p0 + T)
            for name, (us, o) in sdpa_baselines(q, k, v, p0).items():
                mx, _ = err(o, ref)
                print(f"   {name:28s} {us:9.1f} us  {tf / us / 1e6:6.1f} TFLOP/s  max err {mx:.1e}", flush=True)
        us = bench(lambda: attn_prefill(q, k, v, p0), iters=10, warm=2)
        print(f"   attn_prefill                 {us:9.1f} us  {tf / us / 1e6:6.1f} TFLOP/s", flush=True)

    print("== decode timing (per layer, us, two kernels replayed in a CUDA graph)", flush=True)
    k, v = make_cache(16000)
    for T in (1, 2, 4, 8, 12, 16):
        q, _ = make_q(T)
        line = []
        for n in (500, 4000, 12000, 16000):
            nt = torch.tensor([n], dtype=torch.int32, device=dev)
            line.append(f"n={n}: {bench_graph(lambda: attn_decode(q, k, v, nt)):5.1f}")
        us = bench(lambda: attn_decode(q, k, v, nt), iters=50, warm=5)
        print(f"   T={T:2d} graph us  " + "  ".join(line) + f"   (eager n=16000: {us:.1f})", flush=True)
    print("   (KV bytes at n=12000: 49.2 MB = 10.5 us at 4.7 TB/s)", flush=True)


if __name__ == "__main__":
    main()
