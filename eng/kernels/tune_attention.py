"""Tuning experiments for eng.kernels.attention (not a test). Run: python3 -m eng.kernels.tune_attention"""
import sys
import torch
import triton
import triton.language as tl

from . import attention as A
from .test_attention import make_cache, make_q, bench, bench_graph, prefill_flops, dev


@triton.jit
def _empty(X):
    pass


def graph_time(fn, iters=50):
    return bench_graph(fn, iters)


def decode_time(T, n, cfg):
    q, _ = make_q(T)
    k, v = make_cache(16000)
    nt = torch.tensor([n], dtype=torch.int32, device=dev)
    return graph_time(lambda: A.attn_decode(q, k, v, nt, cfg=cfg))


def decode_cold(T, n, cfgs, ncache=12):
    """Per-call time of attn_decode (split + merge, gate and int8 output as in the engine) cycling through
    ncache caches (> the 256 MB last-level cache), as in a forward pass. cfgs: list of cfg dicts."""
    from .test_w4 import graph_time as gtime
    from . import w4a8
    q, gate = make_q(T)
    caches = [((torch.randn(n + 16, 4, 256, device=dev)).bfloat16(), torch.randn(n + 16, 4, 256, device=dev).bfloat16())
              for _ in range(ncache)]
    nt = torch.tensor([n], dtype=torch.int32, device=dev)
    out = torch.empty(T, 24, 256, dtype=torch.bfloat16, device=dev)
    xa = w4a8.new_xa(T, 6144)
    res = []
    for cfg in cfgs:
        try:
            us = gtime([lambda k=k, v=v: A.attn_decode(q, k, v, nt, gate=gate, out=out, cfg=cfg, xa=xa)
                        for k, v in caches]) * 1e6
        except Exception as e:
            print(f"T={T} n={n} {cfg}: failed {type(e).__name__} {str(e)[:200]}", flush=True)
            continue
        res.append((us, cfg))
        print(f"T={T:2d} n={n:5d} {cfg}: {us:6.1f} us  {4 * 256 * 2 * 2 * n / us / 1e6:.2f} TB/s", flush=True)
    res.sort(key=lambda r: r[0])
    print("BEST", T, n, [(round(u, 1), c) for u, c in res[:4]], flush=True)


def hip_cold(T, n, ncache=12):
    from .test_w4 import graph_time as gtime
    from .attention_hip import attn_decode_hip
    q, gate = make_q(T)
    caches = [((torch.randn(n + 16, 4, 256, device=dev)).bfloat16(), torch.randn(n + 16, 4, 256, device=dev).bfloat16())
              for _ in range(ncache)]
    nt = torch.tensor([n], dtype=torch.int32, device=dev)
    out = torch.empty(T, 24, 256, dtype=torch.bfloat16, device=dev)
    ref = A.attn_decode(q, caches[0][0], caches[0][1], nt, gate=gate)
    for ns in (32, 64, 128):
        for st in (2, 3, 4):
            o = attn_decode_hip(q, caches[0][0], caches[0][1], nt, gate=gate, out=out, nsplit=ns, stages=st)
            err = (o.float() - ref.float()).abs().max().item()
            us = gtime([lambda k=k, v=v: attn_decode_hip(q, k, v, nt, gate=gate, out=out, nsplit=ns, stages=st)
                        for k, v in caches]) * 1e6
            print(f"HIP T={T:2d} n={n} nsplit={ns} stages={st}: {us:6.1f} us {4 * 256 * 2 * 2 * n / us / 1e6:.2f} TB/s,"
                  f" max err vs triton {err:.3g}", flush=True)


def main():
    if "hip" in sys.argv:
        for T in (10, 1, 16):
            hip_cold(T, 10000)
            decode_cold(T, 10000, [None])
        return
    if "cold" in sys.argv:
        for T in (10, 1):
            cfgs = [None] + [dict(NSPLIT=ns, BLOCK_N=bn, BLOCK_M=64 if T > 2 else 16, num_warps=nw, num_stages=st,
                                  PARTIAL=torch.bfloat16)
                             for ns in (64, 128, 192, 256) for bn in (16, 32, 64) for nw in (4, 8) for st in (1, 2)]
            for n in (10000,):
                decode_cold(T, n, cfgs)
        return
    x = torch.zeros(1, device=dev)
    print(f"empty triton kernel in graph: {graph_time(lambda: _empty[(1,)](x)):.2f} us", flush=True)
    print(f"empty triton kernel grid 256: {graph_time(lambda: _empty[(256,)](x)):.2f} us", flush=True)
    if "decode" in sys.argv:
        for T in (1, 8, 16):
            for n in (12000,):
                cfgs = [dict(NSPLIT=ns, BLOCK_N=bn, BLOCK_M=bm, num_warps=4, num_stages=st, PARTIAL=pd)
                        for bm in (16, 64) for ns in (16, 32, 48, 64) for bn in (32, 64) for st in (1, 2)
                        for pd in (torch.float32, torch.bfloat16)
                        if bm <= 16 * T and not (bm == 64 and ns < 32) and not (T == 1 and ns < 32)]
                res = []
                for cfg in cfgs:
                    try:
                        us = decode_time(T, n, cfg)
                        res.append((us, cfg))
                        print(f"T={T:2d} n={n:5d} {cfg}: {us:6.1f} us", flush=True)
                    except Exception as e:
                        print(f"T={T} n={n} {cfg}: failed {type(e).__name__} {str(e)[:200]}", flush=True)
                res.sort(key=lambda r: r[0])
                print("BEST", T, n, res[:4], flush=True)
    if "prefill" in sys.argv:
        for T, p0 in [(4096, 6000), (512, 0), (512, 8000), (1024, 0), (12000, 0)]:
            k, v = make_cache(p0 + T)
            q, _ = make_q(T)
            tf = prefill_flops(T, p0)
            for c in [dict(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=2),
                      dict(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2),
                      dict(BLOCK_M=64, BLOCK_N=32, num_warps=4, num_stages=2),
                      dict(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=2, matrix_instr_nonkdim=16)]:
                us = bench(lambda: A.attn_prefill(q, k, v, p0, cfg=c), iters=10, warm=2)
                print(f"prefill T={T} p0={p0} {c}: {us:8.1f} us {tf / us / 1e6:6.1f} TFLOP/s", flush=True)


if __name__ == "__main__":
    main()
