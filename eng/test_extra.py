"""The prompt-augmented draft vocabulary (Fast.set_extra): the patched draft head equals a head quantized from the
same rows, and the per-request setup time. python3 -m eng.test_extra"""
import json
import time

import numpy as np
import torch

from eng import hip as H
from eng.fast import Fast, Linear
from eng.model import Weights


def main():
    w = Weights("/models/Qwen3.8-27B")
    F = Fast(w, 9)
    x = torch.randn(1, 5120, dtype=torch.bfloat16, device="cuda")
    rows = [json.loads(l) for l in open("tests/prompts.jsonl")][:4]
    for r in rows:
        toks = np.asarray(r["prompt_token_ids"], dtype=np.int64)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        F.set_extra(toks)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        ref = Linear(w.lm_head[F.dvocab].contiguous())
        a = torch.empty(1, len(F.dvocab), dtype=torch.bfloat16, device="cuda")
        b = torch.empty_like(a)
        F.dhead(x, a)
        ref(x, b)
        extra = int((~torch.from_numpy(F.in_dv[toks])).sum())
        print(f"{r['id']}: set_extra {dt:.2f} ms, prompt tokens outside dv {extra}, "
              f"slots used {len(np.unique(toks[~F.in_dv[toks]]))}, max |diff| {float((a.float() - b.float()).abs().max())}",
              flush=True)
        # a hidden state along an extra token's head row must draft that token (through the map, as in F.mtp)
        ex = F.dvocab[len(F.dv_np):]
        tokid = int(ex[0])
        h = w.lm_head[tokid:tokid + 1].float()
        F.mhn[:1].copy_((h / h.norm() * 80).to(torch.bfloat16))
        F.dhead(F.mhn[:1], F.mlogit)
        out = torch.zeros(1, dtype=torch.long, device="cuda")
        H.argmax_rows(F.mlogit, out, F.dvocab)
        print(f"  extra token {tokid} -> drafted {int(out)}", flush=True)


if __name__ == "__main__":
    main()
