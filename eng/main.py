import os
import sys
import time

import torch

from .engine import Engine
from .server import serve

MODEL = os.environ.get("ENG_MODEL", "/models/Qwen3.8-27B")


def main():
    port = int(sys.argv[1])
    t0 = time.time()
    eng = Engine(MODEL)
    # warm-up: Triton compiles a kernel variant per integer-argument class (== 1, divisible by 16, other), so run
    # prefill chunks of every class of length and start position, then whole requests (random tokens; the cache
    # is cleared after)
    g = torch.Generator().manual_seed(0)
    rnd = lambda k: torch.randint(1000, 100000, (k,), generator=g).tolist()
    ids = torch.tensor(rnd(4096), device="cuda")
    for p0 in (6400, 6404, 6401):
        for T in (1, 2, 16, 17, 32, 1595, 1600, 4096):
            hn = eng.pre.chunk(ids[:T], p0, eng.st)
            eng.pre.mtp(ids[:T], hn, p0, eng.st)
    base = rnd(12000)
    eng.generate(rnd(8000), 32)
    eng.generate(base, 32)
    for n, keep in ((8000, 6400), (12000, 6400), (8000, 6403), (12000, 6411), (12000, 6000), (8000, 7000)):
        eng.generate(base[:keep] + rnd(n - keep), 32)
    eng.generate([248045, 846, 198, 9707, 248046, 198, 248045, 74455, 198], 4)
    eng.cache.entries.clear()
    eng.cache.in_work = None
    torch.cuda.synchronize()
    print(f"ready in {time.time() - t0:.1f} s", flush=True)
    serve(eng, port)


if __name__ == "__main__":
    main()
