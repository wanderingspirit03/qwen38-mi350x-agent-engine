"""Prefill attention timing over shapes: attn_prefill as dispatched (stream-K for few row blocks over long keys)
against the regular one-program-per-(row block, head) kernel. Run: python3 -m eng.kernels.tune_prefill"""
import torch

from . import attention as A
from .test_attention import make_cache, make_q, bench, prefill_flops


def main():
    for T, p0 in [(4096, 6400), (5600, 6400), (1500, 6400), (1024, 8192), (512, 8000), (300, 12288), (2048, 6400),
                  (1500, 700), (4096, 0)]:
        k, v = make_cache(p0 + T)
        q, gate = make_q(T)
        tf = prefill_flops(T, p0)
        ta = bench(lambda: A.attn_prefill(q, k, v, p0, gate=gate), iters=10, warm=2)
        tr = bench(lambda: A.attn_prefill(q, k, v, p0, gate=gate, cfg=A.prefill_cfg(T)), iters=10, warm=2)
        sk = A._use_stream_k(T, p0, A._num_cus(q.device))
        print(f"T={T:5d} p0={p0:5d}: dispatched {ta:8.1f} us ({tf / ta / 1e6:5.1f} TF/s, {'stream-K' if sk else 'regular'})"
              f"  regular {tr:8.1f} us ({tf / tr / 1e6:5.1f} TF/s)", flush=True)


if __name__ == "__main__":
    main()
