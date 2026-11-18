"""Profile one prefill chunk (T rows at position p0) and print the top kernels by GPU time."""
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

from eng.fast import SeqState
from eng.model import Weights
from eng.prefill import Prefill

T = int(sys.argv[1]) if len(sys.argv) > 1 else 5600
P0 = int(sys.argv[2]) if len(sys.argv) > 2 else 6400
w = Weights("/models/Qwen3.8-27B")
st = SeqState(16384)
pre = Prefill(w, fp8=os.environ.get("ENG_FP8", "1") == "1")
ids = torch.randint(1000, 100000, (T,), device="cuda")
for _ in range(2):
    pre.chunk(ids, P0, st)
torch.cuda.synchronize()
t0 = time.perf_counter()
pre.chunk(ids, P0, st)
torch.cuda.synchronize()
print(f"chunk T={T} p0={P0}: {(time.perf_counter() - t0) * 1e3:.1f} ms")
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    pre.chunk(ids, P0, st)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70))
