"""Error of attn2 on the model's real q/k/v: one 12000-token agentic prompt (attn2_prompt.json, from
tests/qprompts24.jsonl), prefill of 6400 positions, then a 1600-token chunk whose 16 attention inputs are captured.
Prints rel-rms error vs an fp32 reference per layer (mean / worst head) for the bf16 kernel and the given configs.
Run: python3 -m eng.kernels.attn2_real ['[{"RECENT": 8}, {"QK8": false}]']"""
import json
import sys

import torch

import eng.prefill as P
from eng.fast import SeqState
from eng.model import Weights
from . import attention as A
from . import attn2 as B
from .test_attention import ref_attn

CAP = []


def hook(q, k, v, p0, gate=None, out=None):
    T = q.shape[0]
    CAP.append((q.clone(), k[:p0 + T].clone(), v[:p0 + T].clone(), p0, gate.clone()))
    return A.attn_prefill(q, k, v, p0, gate=gate, out=out)


def main():
    variants = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [{}, {"RECENT": 0, "FIRST16": False}]
    ids = json.load(open(__file__.replace("attn2_real.py", "attn2_prompt.json")))
    p0, T = 6400, 1600
    ids = torch.tensor(ids, device="cuda")
    w = Weights("/models/Qwen3.8-27B")
    st = SeqState(16384)
    pre = P.Prefill(w, fp8=True)
    pre.chunk(ids[:p0], 0, st)
    P.attn_prefill = hook
    P.ATTN2 = False
    pre.chunk(ids[p0:p0 + T], p0, st)
    del pre, w
    torch.cuda.empty_cache()
    tot = {}
    for li, (q, k, v, p0, gate) in enumerate(CAP):
        ref = ref_attn(q, k, v, p0 + T) * torch.sigmoid(gate.float())
        outs = [("bf16", A.attn_prefill(q, k, v, p0, gate=gate))]
        outs += [(json.dumps(var), B.attn2_prefill(q, k, v, p0, gate=gate, **var)) for var in variants]
        line = []
        for name, o in outs:
            e = ((o.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
            eh = ((o.float() - ref).pow(2).mean((0, 2)).sqrt() / ref.pow(2).mean((0, 2)).sqrt()).max().item()
            tot.setdefault(name, []).append((e, eh))
            line.append(f"{e:.2e}/{eh:.2e}")
        print(f"layer {li:2d}: " + "  ".join(line), flush=True)
    for name, es in tot.items():
        print(f"{name:40s} mean relrms {sum(e for e, _ in es) / len(es):.2e}  max head {max(h for _, h in es):.2e}")


if __name__ == "__main__":
    main()
