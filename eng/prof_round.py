"""Per-kernel breakdown of the decode round, from HIP graph replays (torch profiler): kernel time per category,
the round's wall time, and the gaps (wall time not covered by any kernel).
python3 -m eng.prof_round [reps]"""
import os
import re
import sys
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from eng.bench_round import graph, tm
from eng.fast import Fast, SeqState
from eng.model import Weights

K = int(os.environ.get("ENG_SPEC_K", "9"))
N = int(os.environ.get("BENCH_POS", "10000"))

CATS = [("GEMM w4a8", r"w4a8_kernel"), ("quant_x", r"quant_x"), ("GEMM w4", r"w4_gemm|w8"),
        ("GDN verify/commit", r"gdn_verify|gdn_commit|conv_commit"), ("GDN prep", r"gdn_prep"),
        ("gated norm", r"gated_norm"), ("attention", r"_decode_|attn"), ("norms", r"rmsnorm"),
        ("argmax", r"argmax"), ("mtp_in", r"mtp_in"), ("accept", r"accept"), ("torch/copies", r".")]


def cat(name):
    for c, pat in CATS:
        if re.search(pat, name):
            return c


def breakdown(g, reps, label):
    torch.cuda.synchronize()
    wall = tm(g.replay)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
    ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    ev.sort(key=lambda e: e.time_range.start)
    per_cat = defaultdict(float)
    per_k = defaultdict(lambda: [0.0, 0])
    busy = 0.0
    end = None
    for e in ev:
        d = e.time_range.elapsed_us()
        per_cat[cat(e.name)] += d
        per_k[e.name][0] += d
        per_k[e.name][1] += 1
        s, t = e.time_range.start, e.time_range.end
        if end is None or s >= end:
            busy += t - s
            end = t
        elif t > end:
            busy += t - end
            end = t
    span = (ev[-1].time_range.end - ev[0].time_range.start) / reps if ev else 0
    print(f"== {label}: wall {wall * 1e3:.0f} us per replay (timed), traced span {span:.0f} us, kernels busy "
          f"{busy / reps:.0f} us, kernels {len(ev) / reps:.0f} per replay", flush=True)
    for c, v in sorted(per_cat.items(), key=lambda kv: -kv[1]):
        print(f"   {c:20s} {v / reps:8.1f} us", flush=True)
    for name, (v, n) in sorted(per_k.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"   {v / reps:8.1f} us  {n // reps:4d} x {v / n:6.1f} us  {name[:90]}", flush=True)
    return wall


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    w = Weights("/models/Qwen3.8-27B")
    st = SeqState(16384)
    F = Fast(w, K)
    F.vt.fill_(100)
    parts = {
        "verify": lambda: F.verify(st),
        "accept+commit": lambda: F.accept(st),
        "mtp catch-up": lambda: F.mtp(st, F.new, F.hn, F.P[0:1], F.P[1:2], F.R, F.zero, F.vt[1:2]),
        "mtp 1 row": lambda: F.mtp(st, F.vt[:1], F.mhn[:1], F.P[3:4], F.P[4:5], 1, F.zero, F.vt[1:2]),
    }
    for name, fn in parts.items():
        F.set_pos(N)
        F.P[:2] = N
        F.n_acc.fill_(K + 1)
        g = graph(fn)
        F.set_pos(N)
        breakdown(g, reps, name)
    F.capture(st)
    F.set_pos(N)
    breakdown(F.g_round, reps, f"whole round K={K}")


if __name__ == "__main__":
    main()
