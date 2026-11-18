#!/usr/bin/env python3
"""tests/client.py PORT PROMPTS.jsonl [N] [MAX_TOKENS]: send prompts one at a time like the judge, print timing
and the answers' token ids (JSON lines prefixed 'ANS ')."""
import http.client
import json
import sys
import time

port, path = int(sys.argv[1]), sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
mt = int(sys.argv[4]) if len(sys.argv) > 4 else 128
rows = [json.loads(l) for l in open(path)][:n]
tot_tok = tot_s = 0
for r in rows:
    body = {"model": "x", "prompt": r["prompt_token_ids"], "max_tokens": mt, "temperature": 0.0, "logprobs": 5,
            "return_tokens_as_token_ids": True, "stream": False}
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    t0 = time.monotonic()
    c.request("POST", "/v1/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
    resp = c.getresponse()
    doc = json.loads(resp.read())
    dt = time.monotonic() - t0
    ch = doc["choices"][0]
    toks = ch["token_ids"]
    k = len(toks) - (1 if ch["finish_reason"] == "stop" else 0)
    tot_tok += k
    tot_s += dt
    print(f"{r.get('id')} prompt={len(r['prompt_token_ids'])} out={len(toks)} {ch['finish_reason']} {dt:.3f}s "
          f"{k / dt:.1f} tok/s", flush=True)
    print("ANS " + json.dumps({"id": r.get("id"), "tokens": toks}), flush=True)
print(f"TOTAL {tot_tok} tokens {tot_s:.2f} s -> {tot_tok / tot_s:.2f} tok/s")
