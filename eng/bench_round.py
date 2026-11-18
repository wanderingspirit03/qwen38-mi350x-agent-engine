"""Time the decode round and its parts as HIP graphs (state contents are garbage; shapes and lengths are real)."""
import os
import time

import torch

from eng.fast import Fast, SeqState
from eng.model import Weights

K = int(os.environ.get("ENG_SPEC_K", "4"))
N = int(os.environ.get("BENCH_POS", "10000"))


def tm(f, n=30):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def main():
    w = Weights("/models/Qwen3.8-27B")
    st = SeqState(16384)
    F = Fast(w, K)
    F.vt.fill_(100)
    parts = {
        "verify": lambda: F.verify(st),
        "accept+commit": lambda: F.accept(st),
        "mtp catch-up": lambda: F.mtp(st, F.new, F.hn, F.pos, F.R, F.zero),
        "mtp 1 row": lambda: F.mtp(st, F.vt[:1], F.mhn[:1], F.mpos, 1, F.zero),
    }
    for name, fn in parts.items():
        F.pos.fill_(N)
        F.mpos.fill_(N)
        F.n_acc.fill_(K + 1)
        g = graph(fn)
        F.pos.fill_(N)
        print(f"{name}: {tm(g.replay):.3f} ms", flush=True)
    F.capture(st)
    def rnd():
        F.pos.fill_(N)
        F.g_round.replay()
    print(f"whole round (K={K}): {tm(rnd):.3f} ms", flush=True)


if __name__ == "__main__":
    main()
