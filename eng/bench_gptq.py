"""Time of eng.gptq.gptq_codes per target matrix shape (random weights, H from random activations).
python3 -m eng.bench_gptq"""
import time

import torch

from .gptq import gptq_codes


def main():
    for N, K in ((34816, 5120), (5120, 17408), (16480, 5120), (5120, 6144), (14336, 5120)):
        W = torch.randn(N, K, device="cuda") * 0.02
        X = torch.randn(4096, K, device="cuda")
        H = X.t() @ X
        torch.cuda.synchronize()
        t0 = time.time()
        gptq_codes(W, H)
        torch.cuda.synchronize()
        print(f"gptq {N}x{K}: {time.time() - t0:.2f} s", flush=True)


if __name__ == "__main__":
    main()
