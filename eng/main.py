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
    # warm-up: compile every kernel shape the judged prompt sizes use (random prompts; cache cleared after)
    g = torch.Generator().manual_seed(0)
    for n in (8000, 12000, 3000, 500):
        ids = torch.randint(1000, 100000, (n,), generator=g).tolist()
        eng.generate(ids, 32)
        eng.generate(ids[:n // 2] + torch.randint(1000, 100000, (n - n // 2,), generator=g).tolist(), 32)
    eng.generate([248045, 846, 198, 9707, 248046, 198, 248045, 74455, 198], 4)
    eng.cache.entries.clear()
    eng.cache.in_work = None
    torch.cuda.synchronize()
    print(f"ready in {time.time() - t0:.1f} s", flush=True)
    serve(eng, port)


if __name__ == "__main__":
    main()
