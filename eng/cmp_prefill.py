"""Compare the fast prefill path with the plain v0 forward, layer by layer, on a real prompt prefix."""
import json
import sys

import torch

from eng import model as M
from eng.fast import SeqState
from eng.prefill import Prefill

T = int(sys.argv[1]) if len(sys.argv) > 1 else 64
torch.manual_seed(0)
w = M.Weights("/models/Qwen3.8-27B")
m = M.Model.__new__(M.Model)
m.w, m.device = w, "cuda"
ids = torch.randint(0, 200000, (T,), device="cuda")

# v0 with per-layer capture
st0 = M.State(4096)
outs0 = []
orig_layers = w.layers
with torch.inference_mode():
    for nl in (1, 2, 3, 4, 8, 64):
        w.layers = orig_layers[:nl]
        st0.reset()
        _, hn0 = m.forward(ids, 0, st0)
        st1 = SeqState(4096)
        pre = Prefill(w)
        hn1 = pre.chunk(ids, 0, st1).clone()
        torch.cuda.synchronize()
        d = (hn0.float() - hn1.float()).abs()
        print(f"layers {nl}: max diff {d.max().item():.4f} mean {d.mean().item():.5f} (|h| mean {hn0.float().abs().mean().item():.4f})", flush=True)
