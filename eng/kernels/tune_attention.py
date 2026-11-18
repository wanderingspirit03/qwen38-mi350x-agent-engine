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


def main():
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
