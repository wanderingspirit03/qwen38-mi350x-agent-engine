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
    eng.generate([248045, 846, 198, 9707, 248046, 198, 248045, 74455, 198], 4)   # warm-up
    torch.cuda.synchronize()
    print(f"ready in {time.time() - t0:.1f} s", flush=True)
    serve(eng, port)


if __name__ == "__main__":
    main()
