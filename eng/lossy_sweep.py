"""Lossy-acceptance sweep: python3 -m eng.lossy_sweep CONFIGS PROMPTS.jsonl... (ENG_SPEC_K sets K).
CONFIGS: comma list of tau[:price] (tau 0: exact matches only; see accept_kernel). One engine answers every prompt
(at most 128 tokens) under each config and prints the answer tokens per round and the lossy tokens' added margin per
token (the target's own estimate of the added mean gap)."""
import json
import sys

from .engine import SPEC_K, Engine
from .fast import lossy_params
from .main import MODEL

cfgs = [tuple(float(y) for y in x.split(":")) for x in sys.argv[1].split(",")]
rows = [json.loads(l) for p in sys.argv[2:] for l in open(p)]
eng = Engine(MODEL)
for cfg in cfgs:
    lp = eng.fast.lossy
    lp.copy_(lp.new_tensor(lossy_params(*cfg) + [0.0, 0.0]))
    tok = rnd = tok_l = rnd_l = 0
    for r in rows:
        out, fin = eng.generate(r["prompt_token_ids"], 128)
        tok += len(out)
        rnd += len(eng.accepted)
        if fin == "length":     # the judge's answers all run to 128 tokens
            tok_l += len(out)
            rnd_l += len(eng.accepted)
    msum, cnt = lp[2].item(), lp[3].item()
    print(f"K={SPEC_K} tau:price={':'.join(map(str, cfg))}: {tok / rnd:.3f} tokens per round ({tok} in {rnd}); "
          f"{tok_l / max(rnd_l, 1):.3f} on answers of 128 ({rnd_l} rounds); lossy {cnt:.0f} ({cnt / tok:.3f} per token), added margin {msum / tok:.4f} per token, "
          f"{(msum / cnt if cnt else 0):.3f} per lossy token", flush=True)
