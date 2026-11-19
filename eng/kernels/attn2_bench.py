"""Timing (and a quick accuracy check) of attn2 configs against attention.attn_prefill; best of 3 x 20 calls.
Run: python3 -m eng.kernels.attn2_bench [run.json with {"cfgs": [...], "shapes": [[T, p0], ...]}]"""
import json
import sys
import torch
from . import attention as A
from . import attn2 as B
from .test_attention import make_cache, make_q, bench, ref_attn


def flops(T, p0):
    return 4 * 24 * 256 * (T * p0 + T * (T + 1) / 2)


def relrms(o, ref):
    return ((o.float() - ref).pow(2).mean() / ref.pow(2).mean()).sqrt().item()


def main():
    torch.manual_seed(0)
    run = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    cfgs = run.get("cfgs", [{}])
    shapes = run.get("shapes", [(1024, 6400), (2048, 6400), (4096, 6400), (5600, 6400)])
    for T, p0 in [(700, 3000), (300, 0), (129, 70), (1500, 6400), (1, 5000), (200, 9000)]:
        k, v = make_cache(p0 + T)
        q, gate = make_q(T)
        ref = ref_attn(q, k, v, p0 + T) * torch.sigmoid(gate.float())
        line = f"T={T} p0={p0} relrms base {relrms(A.attn_prefill(q, k, v, p0, gate=gate), ref):.2e}"
        for c in cfgs:
            try:
                line += f" | {relrms(B.attn2_prefill(q, k, v, p0, gate=gate, **c), ref):.2e}"
            except Exception as e:
                line += f" | {type(e).__name__}"
        print(line, flush=True)
    for T, p0 in shapes:
        k, v = make_cache(p0 + T)
        q, gate = make_q(T)
        tf = flops(T, p0)
        fns = [("base", lambda: A.attn_prefill(q, k, v, p0, gate=gate))]
        for c in cfgs:
            fns.append((json.dumps(c), lambda c=c: B.attn2_prefill(q, k, v, p0, gate=gate, **c)))
        best = {}
        for rep in range(3):
            for name, fn in fns:
                try:
                    t = bench(fn, iters=20, warm=2)
                except Exception as e:
                    t = float("nan")
                best[name] = min(best.get(name, t), t)
        t0 = best["base"]
        print(f"T={T} p0={p0} base {t0:7.1f} us {tf / t0 / 1e6:5.0f} TF/s", flush=True)
        for name, _ in fns[1:]:
            t = best[name]
            print(f"   {name:60s} {t:7.1f} us {tf / t / 1e6:5.0f} TF/s  x{t0 / t:.2f}", flush=True)


if __name__ == "__main__":
    main()
