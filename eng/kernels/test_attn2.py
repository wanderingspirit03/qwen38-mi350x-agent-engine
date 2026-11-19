"""Accuracy test of eng.kernels.attn2 against the fp32 reference (random data).
Run: ./try tests/in_container.sh python3 -m eng.kernels.test_attn2"""
import torch

from . import attn2 as B
from .test_attention import make_cache, make_q, ref_attn

SHAPES = [(1, 0), (1, 5000), (7, 63), (64, 0), (129, 70), (200, 9000), (300, 0), (511, 1), (700, 3000), (1024, 6400),
          (1500, 6400), (2048, 0), (3000, 2500), (4096, 6400)]


def main():
    torch.manual_seed(0)
    bad = 0
    for T, p0 in SHAPES:
        k, v = make_cache(p0 + T)
        q, gate = make_q(T)
        ref = ref_attn(q, k, v, p0 + T)
        refg = ref * torch.sigmoid(gate.float())
        line = f"T={T:5d} p0={p0:5d}:"
        for cfg in [dict(), dict(QK8=False), dict(RECENT=0, FIRST16=False), dict(RECENT=100000), dict(gate=False)]:
            cfg = dict(cfg)
            g = None if cfg.pop("gate", True) is False else gate
            out = torch.full((T, 24, 512), float("nan"), dtype=torch.bfloat16, device="cuda")[..., :256]
            o = B.attn2_prefill(q, k, v, p0, gate=g, out=out, **cfg)
            r = refg if g is not None else ref
            e = ((o.float() - r).pow(2).mean() / r.pow(2).mean()).sqrt().item()
            lim = 3e-3 if cfg.get("RECENT") == 100000 else 6e-2
            o2 = B.attn2_prefill(q, k, v, p0, gate=g, **cfg)
            same = torch.equal(o, o2)
            ok = e == e and e < lim and same
            bad += not ok
            line += f" {e:.1e}{'' if ok else '!'}{'' if same else '(nondet)'}"
        print(line, flush=True)
    print("FAILED" if bad else "all passed", bad)


if __name__ == "__main__":
    main()
