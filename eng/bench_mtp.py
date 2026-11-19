"""Decompose the MTP draft chain (Fast.draft_chain: K - 1 single-row MTP steps): time per step in a HIP graph, the
kernel list of one step from a profiler trace of the graph replay (duration, gap before), and the weight bytes each
GEMM streams. python3 -m eng.bench_mtp [K]"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from eng import fast as FM
from eng.fast import EXTRA, Fast, Linear, SeqState
from eng.kernels import w4 as w4q
from eng.model import HEADS, HEAD_DIM, HIDDEN, INTER, Weights

K = int(sys.argv[1]) if len(sys.argv) > 1 else 9
dev = "cuda"
w = Weights("/models/Qwen3.8-27B")
F = Fast.__new__(Fast)
F.w, F.K, F.R = w, K, K + 1
R = F.R
M = w.mtp
F.mlin = Fast._layer_lin(M, True, Linear)
F.mlin["fc"] = Linear(M["fc"])
dv = np.load(os.path.join(os.path.dirname(FM.__file__), "draft_vocab.npy")).astype(np.int64)
dv = np.concatenate([dv, np.full(EXTRA, dv[0])])
F.dvocab = torch.from_numpy(dv).to(dev)
F.dhead = Linear(w.lm_head[F.dvocab].contiguous())
bf = dict(dtype=torch.bfloat16, device=dev)
F.x = torch.zeros(R, HIDDEN, **bf)
F.qkv = torch.zeros(R, 14336, **bf)
F.q = torch.zeros(R, HEADS, HEAD_DIM, **bf)
F.ao = torch.zeros(R, HEADS * HEAD_DIM, **bf)
F.act = torch.zeros(R, INTER, **bf)
F.mx = torch.zeros(R, 2 * HIDDEN, **bf)
F.mh = torch.zeros(R, HIDDEN, **bf)
F.mhn = torch.randn(R, HIDDEN, **bf)
F.mlogit = torch.zeros(1, len(dv), **bf)
F.am = w4q.argmax_slots(dev)
F.aoq = {}
F.vt = torch.randint(0, 30000, (R,), device=dev)
F.P = torch.full((3 + 2 * K,), 5000, dtype=torch.int32, device=dev)
F.zero = torch.zeros(1, dtype=torch.long, device=dev)
st = SeqState(8192)
st.k.normal_()
st.v.normal_()

for k, lin in list(F.mlin.items()) + [("dhead", F.dhead)]:
    p = lin.p
    mb = (p["qw"].numel() * p["qw"].element_size() + p["meta"].numel() * p["meta"].element_size()) / 1e6
    print(f"{k:8s} N={p['N']:6d} K={p['K']:6d} {mb:6.1f} MB", flush=True)

F.draft_chain(st)
torch.cuda.synchronize()
g_chain = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_chain):
    F.draft_chain(st)
g_step = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_step):
    F.mtp(st, F.vt[1:2], F.mhn[:1], F.P[3:4], F.P[4:5], 1, F.zero, F.vt[2:3], decode=False)
torch.cuda.synchronize()


def timeit(f, n=200):
    for _ in range(10):
        f()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


tc = timeit(g_chain.replay)
ts = timeit(g_step.replay)
print(f"chain of {K - 1} steps: {tc:.1f} us ({tc / (K - 1):.1f} us per step); one step graph replayed: {ts:.1f} us",
      flush=True)

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        g_chain.replay()
    torch.cuda.synchronize()
path = "/tmp/mtp_trace.json"
prof.export_chrome_trace(path)
ev = [e for e in json.load(open(path))["traceEvents"] if e.get("cat") == "kernel"]
ev.sort(key=lambda e: e["ts"])
print(f"{len(ev)} kernels traced over 3 chains", flush=True)
n = len(ev) // 3
chain = ev[n:2 * n]          # the middle replay
per = n // (K - 1)
print(f"{per} kernels per step; middle step of the middle chain:")
s0 = (K - 1) // 2 * per
prev = chain[s0 - 1]
tot_d = tot_g = 0
for e in chain[s0:s0 + per]:
    gap = e["ts"] - (prev["ts"] + prev["dur"])
    tot_d += e["dur"]
    tot_g += gap
    print(f"  {e['dur']:7.1f} us  gap {gap:5.1f}  {e['name'][:90]}")
    prev = e
print(f"  sum kernels {tot_d:.1f} us, sum gaps {tot_g:.1f} us", flush=True)
# per-kernel average over all steps of the middle chain
names = {}
for j, e in enumerate(chain):
    names.setdefault(j % per, []).append(e["dur"])
print("mean over the chain's steps (us):", [round(sum(v) / len(v), 1) for _, v in sorted(names.items())])
