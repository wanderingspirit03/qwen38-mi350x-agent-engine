"""After the serve warm-up, run prompts of random lengths and shared-prefix lengths; print slow prefills."""
import time
import numpy as np
import torch
from .engine import Engine
from .main import MODEL, warm_up

eng = Engine(MODEL)
warm_up(eng)
rng = np.random.default_rng(1)
base = rng.integers(1000, 100000, 12000).tolist()
for i in range(40):
    n = int(rng.integers(6500, 12000))
    keep = int(rng.integers(6380, 6420)) if i % 5 else 0
    ids = (base[:keep] + rng.integers(1000, 100000, n - keep).tolist())
    t0 = time.monotonic()
    eng.generate(ids, 8)
    print(f"probe {i}: n={n} keep={keep} {1e3 * (time.monotonic() - t0):.0f} ms", flush=True)
