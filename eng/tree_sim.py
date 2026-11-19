"""Tree speculation feasibility, step 1: teacher-forced MTP draft ranks along the true continuation.

For each prompt: the engine generates MAXT tokens (greedy; its output is the truth); the target re-reads
prompt + output (prefill path) to get the final-normed hidden per position and the MTP KV; then for every anchor p
(x = full[p]) the MTP chain is run on the TRUE tokens (catch-up step from the target hidden at p-1, then chain
steps from the MTP hidden), D depths, and the rank of the true token full[p+j] in the draft-vocab logits of depth j
is recorded (255 = not in the draft vocab / past the end), with the probability of the true token and of the top-1.
A tree's matched path always follows the true tokens, so these ranks decide the accept length of any static tree.
Output: lines 'SIM <json>' with base64 uint8 arrays (ranks, ptrue*255, ptop*255) and the engine's round sizes.
python3 -m eng.tree_sim [N_PROMPTS] [MAXT] [D] [FIRST_PROMPT]
"""
import base64
import json
import os
import sys
import zlib

import numpy as np
import torch

from .engine import Engine
from .main import MODEL
from .prefill import CHUNK


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


@torch.inference_mode()
def main():
    os.set_blocking(1, True)            # a non-blocking stdout pipe raised BlockingIOError on the long lines
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    S0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    MAXT = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    D = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    eng = Engine(MODEL)
    F, st, pre = eng.fast, eng.st, eng.pre
    V = eng.w.lm_head.shape[0]
    inv = torch.full((V,), -1, dtype=torch.long, device="cuda")
    inv[F.dvocab] = torch.arange(len(F.dvocab), device="cuda")
    rows = [json.loads(l) for l in open("tests/prompts.jsonl")][S0:N]
    pos = torch.zeros(1, dtype=torch.int32, device="cuda")
    nkv = torch.zeros(1, dtype=torch.int32, device="cuda")
    dummy = torch.zeros(1, dtype=torch.long, device="cuda")
    from .fast import Linear
    fhead = Linear(eng.w.lm_head)                       # 4-bit full-vocabulary head (same rows as the draft head)
    flog = torch.empty(1, V, dtype=torch.bfloat16, device="cuda")
    NT = 16
    for r in rows:
        prompt = r["prompt_token_ids"]
        n = len(prompt)
        out, _ = eng.generate(prompt, MAXT)
        acc = list(eng.accepted)
        full = prompt + out
        L = len(full)
        ids = torch.tensor(full + [0] * (D + 2), device="cuda")
        st.ssm.zero_()
        st.conv.zero_()
        eng.cache.in_work = None
        HN = torch.empty(L, 5120, dtype=torch.bfloat16, device="cuda")
        for s in range(0, L, CHUNK):
            b = min(L, s + CHUNK)
            hn = pre.chunk(ids[s:b], s, st)
            HN[s:b] = hn
            m = min(b, L - 1) - s
            pre.mtp(ids[s + 1:s + 1 + m], hn[:m], s, st)
        ks = st.k[16, n - 1:L + D].clone()
        vs = st.v[16, n - 1:L + D].clone()
        tinv = inv[ids]                                 # draft-vocab index of every token (-1: not in it)
        A = L - 1 - n                                   # anchors p = n .. L-2
        rk = torch.full((A, D), 255, dtype=torch.long, device="cuda")
        pt = torch.zeros(A, D, device="cuda")
        p1 = torch.zeros(A, D, device="cuda")
        top = torch.zeros(A, D, NT, dtype=torch.int32, device="cuda")   # full-vocabulary top-NT ids per step
        for a in range(A):
            p = n + a
            for j in range(1, D + 1):
                if p + j >= L:
                    break
                pos.fill_(p + j - 2)
                nkv.fill_(p + j - 1)
                hid = HN[p - 1:p] if j == 1 else F.mhn[:1]
                F.mtp(st, ids[p + j - 1:p + j], hid, pos, nkv, 1, F.zero, dummy)
                lg = F.mlogit[0].float()
                ti = tinv[p + j]
                tv = lg[ti.clamp(min=0)]
                rk[a, j - 1] = torch.where(ti >= 0, (lg > tv).sum().clamp(max=254), 255)
                lse = torch.logsumexp(lg, 0)
                pt[a, j - 1] = torch.where(ti >= 0, (tv - lse).exp(), 0.0)
                p1[a, j - 1] = (lg.max() - lse).exp()
                fhead(F.mhn[:1], flog)
                top[a, j - 1] = flog[0].topk(NT).indices.int()
            st.k[16, n - 1:L + D] = ks
            st.v[16, n - 1:L + D] = vs
        rk = rk.cpu().numpy()
        chain = 0
        rr = 0
        p = 0
        while p < A:
            k = 0
            while k < 8 and rk[p, k] == 0:
                k += 1
            chain += k + 1
            rr += 1
            p += k + 1
        print(f"{r.get('id')} n={n} gen={L - n} engine {sum(acc) / len(acc):.2f} tok/round, sim chain K=8 "
              f"{chain / rr:.2f}", flush=True)
        doc = {"id": r.get("id"), "n": n, "A": A, "D": D, "acc": acc, "rk": b64(rk),
               "pt": b64((pt.cpu().numpy() * 255).round()), "p1": b64((p1.cpu().numpy() * 255).round()),
               "out": out, "top": base64.b64encode(zlib.compress(top.cpu().numpy().tobytes())).decode()}
        line = json.dumps(doc)
        for i in range(0, len(line), 4000):            # in pieces: 'SIMP' parts, then 'SIME' ends the document
            print("SIMP " + line[i:i + 4000], flush=True)
        print("SIME", flush=True)


if __name__ == "__main__":
    main()
