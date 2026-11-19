"""Training / evaluation data for the MTP draft (eng.mtp_train): the engine's greedy answers to
PROMPTS[START:START+N] (up to MAXT tokens) and the target's final hidden (fp8 prefill path, as eng.draft_study) over
prompt + answer, saved to the development store (tests/store.sh) in shards TAG-<k>.pt of 64 sequences:
{"ids": [int32 (L,)], "n": [prompt length], "hn": [bf16 (L - SYS, HIDDEN)]}; the hidden of the shared system prompt
(positions < SYS) once, as sys.pt.
python3 -m eng.mtp_data PROMPTS.jsonl START N MAXT TAG"""
import json
import sys
import time

import torch

from . import prefill as PF
from . import store
from .engine import Engine
from .main import MODEL
from .model import HIDDEN

SYS = 5957


@torch.inference_mode()
def main():
    path, start, N, maxt, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    rows = [json.loads(l)["prompt_token_ids"] for l in open(path)][start:start + N]
    eng = Engine(MODEL)
    t0 = time.time()
    outs = []
    for p in rows:
        out, _ = eng.generate(p, maxt)
        outs.append(out[:maxt])
    print(f"generated {sum(map(len, outs))} tokens for {len(rows)} prompts in {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    st, pre = eng.st, eng.pre
    eng.cache.entries.clear()
    eng.cache.in_work = None
    st.ssm.zero_()
    st.conv.zero_()
    sys_ids = torch.tensor(rows[0][:SYS], device="cuda")
    assert all(p[:SYS] == rows[0][:SYS] for p in rows)
    hs = pre.chunk(sys_ids, 0, st).clone()
    if "sys.pt" not in store.names():
        store.save("sys.pt", {"ids": sys_ids.int().cpu(), "hn": hs.cpu()})
    ssm0, conv0 = st.ssm.clone(), st.conv.clone()
    shard, k = {"ids": [], "n": [], "hn": []}, 0
    for i, (p, out) in enumerate(zip(rows, outs)):
        ids = torch.tensor(p + out, device="cuda")
        L = ids.shape[0]
        st.ssm.copy_(ssm0)
        st.conv.copy_(conv0)
        hn = torch.empty(L - SYS, HIDDEN, dtype=torch.bfloat16, device="cuda")
        for s in range(SYS, L, PF.CHUNK):
            b = min(L, s + PF.CHUNK)
            hn[s - SYS:b - SYS] = pre.chunk(ids[s:b], s, st)
        shard["ids"].append(ids.int().cpu())
        shard["n"].append(len(p))
        shard["hn"].append(hn.cpu())
        if len(shard["n"]) == 64 or i == len(rows) - 1:
            store.save(f"{tag}-{k}.pt", shard)
            shard, k = {"ids": [], "n": [], "hn": []}, k + 1
    print(f"hidden of {len(rows)} sequences saved in {k} shards in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
