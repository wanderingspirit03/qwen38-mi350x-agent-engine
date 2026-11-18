"""Which call is slow on a new prefill length after warm-up? cProfile of the first chunk call."""
import cProfile, pstats, time
import torch
from .engine import Engine
from .main import MODEL, warm_up

eng = Engine(MODEL)
warm_up(eng)
ids = torch.randint(1000, 100000, (8192,), device="cuda")
for T, p0 in ((972, 6144), (1995, 6144), (2011, 6402), (528, 10240), (972, 6144)):
    torch.cuda.synchronize()
    pr = cProfile.Profile()
    t0 = time.monotonic()
    pr.enable()
    hn = eng.pre.chunk(ids[:T], p0, eng.st)
    eng.pre.mtp(ids[:T], hn, p0, eng.st)
    torch.cuda.synchronize()
    pr.disable()
    dt = time.monotonic() - t0
    print(f"=== T={T} p0={p0}: {dt * 1e3:.0f} ms", flush=True)
    if dt > 0.3:
        pstats.Stats(pr).sort_stats("cumulative").print_stats(25)
