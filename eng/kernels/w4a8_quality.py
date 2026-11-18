"""W4A8 quality study: per-linear output error of 4-bit weights with bf16 x (current w4) and with int8 per-group x,
on real activations of a real prompt through the bf16 reference model.
python3 -m eng.kernels.w4a8_quality [T] [layers]"""
import json
import sys

import torch
import torch.nn.functional as F

from .. import model as MD
from .. import ref_ops as R
from . import w4

T = int(sys.argv[1]) if len(sys.argv) > 1 else 128
LAYERS = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3, 7, 15, 23, 31, 39, 47, 55, 59, 62, 63]


def q8(x, g=128):
    """symmetric int8 per (row, group of g) -> dequantized fp32"""
    M, K = x.shape
    xf = x.float().reshape(M, K // g, g)
    s = xf.abs().amax(-1, keepdim=True).clamp(min=1e-20) / 127
    return ((xf / s).round().clamp(-127, 127) * s).reshape(M, K)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def rowrel(a, b):
    return ((a - b).norm(dim=-1) / b.norm(dim=-1))


last = {}
orig_rms = R.rmsnorm


def rms_hook(x, w, eps=1e-6):
    y = orig_rms(x, w, eps)
    last["x"], last["w"], last["y"] = x, w, y
    return y


R.rmsnorm = rms_hook
MD.R.rmsnorm = rms_hook
names = {}
results = []
orig_linear = F.linear


def lin_hook(x, W, b=None):
    y = orig_linear(x, W, b)
    nm = names.get(id(W))
    if nm is None or x.dim() != 2:
        return y
    with torch.no_grad():
        yref = x.float() @ W.float().t()
        fused = last.get("y") is x
        if fused:
            xr, nw = last["x"], last["w"]
            r = torch.rsqrt(xr.float().pow(2).mean(-1, keepdim=True) + 1e-6)
            p = w4.quantize(W, norm_weight=nw)
            wd = w4.dequantize(p)
            y4 = (xr.float() @ wd.t()) * r
            y48 = (q8(xr) @ wd.t()) * r
            y48g64 = (q8(xr, 64) @ wd.t()) * r
            pn = w4.quantize(W)
            y48n = q8(x) @ w4.dequantize(pn).t()        # quantize the normed x instead
            xa = xr
        else:
            p = w4.quantize(W)
            wd = w4.dequantize(p)
            y4 = x.float() @ wd.t()
            y48 = q8(x) @ wd.t()
            y48g64 = q8(x, 64) @ wd.t()
            y48n = y48
            xa = x
        ya8 = q8(x) @ W.float().t()
        xf = xa.float()
        rr = rowrel(y48, yref)
        res = dict(name=nm, fused=fused, w4=rel(y4, yref), w4a8=rel(y48, yref), w4a8_g64=rel(y48g64, yref),
                   w4a8_normedx=rel(y48n, yref), a8only=rel(ya8, yref), w4a8_vs_w4=rel(y48, y4),
                   row_max=rr.max().item(), peak=(xf.abs().amax(-1) / xf.pow(2).mean(-1).sqrt()).max().item())
        results.append(res)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
        del p, wd
    return y


F.linear = lin_hook
torch.nn.functional.linear = lin_hook
MD.F.linear = lin_hook

m = MD.Model("/models/Qwen3.8-27B")
for i, L in enumerate(m.w.layers):
    if i in LAYERS:
        for k in ("gate_up", "down", "qkv", "o", "in_proj", "out"):
            if k in L:
                names[id(L[k])] = f"L{i}.{k}"
import os
prompts = json.load(open(os.path.join(os.path.dirname(__file__), "w4a8_prompts.json")))
for ids in prompts[:2]:
    st = MD.State(4096)
    m.forward(torch.tensor(ids[:T], device="cuda"), 0, st)
import collections
agg = collections.defaultdict(list)
for r in results:
    agg[r["name"].split(".")[1]].append(r)
for k, v in agg.items():
    print(k, " ".join(f"{f}={sum(r[f] for r in v) / len(v):.4f}" for f in
                      ("w4", "w4a8", "w4a8_g64", "w4a8_normedx", "a8only", "w4a8_vs_w4", "row_max", "peak")), flush=True)
