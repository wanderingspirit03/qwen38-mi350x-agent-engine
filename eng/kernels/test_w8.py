"""8-bit head (w4.quantize8): correctness, error vs bf16 next to the 4-bit head, speed.
python3 -m eng.kernels.test_w8 [check|error|tune]"""
import sys
import torch
from . import w4
from .test_w4 import load, rel, graph_time, make_packed, TUNE_CFGS

HEADS = {"lm_head": 248320, "dhead": 32768}


def check(N=4096, K=2048):
    w = torch.randn(N, K, device="cuda").bfloat16()
    nw = (torch.randn(K, device="cuda") * 0.3).bfloat16()
    bad = n = 0
    for norm in (None, nw):
        p = w4.quantize8(w, norm_weight=norm)
        wf = w.float() * ((1 + nw.float())[None] if norm is not None else 1)
        print(f"norm={norm is not None}: weight error of quantize8 {rel(w4.dequantize8(p), wf):.2e}", flush=True)
        for M in (1, 3, 9, 16):
            x = (torch.randn(M, K, device="cuda") * 3).bfloat16()
            yr = w4.reference(x, p)
            for cfg in [(nt, wv, ks) for nt in (1, 2, 4, 8) for wv in (1, 2, 4, 8, 16) for ks in (1, 3)]:
                if (cfg[1] == 16 and cfg[0] > 2) or w4.lds_bytes(cfg[0], cfg[2], 2 * K, 128) > 65536:
                    continue
                e = rel(w4.gemm(x, p, cfg=cfg), yr)
                n += 1
                if e > 5e-3:
                    bad += 1
                    print(f"FAIL norm={norm is not None} M={M} cfg={cfg} err {e:.2e}", flush=True)
    print(f"quantize8 gemm: {bad} failures of {n}", flush=True)


def error():
    W = load(["lm_head.weight"]).bfloat16()
    torch.manual_seed(0)
    for name, N in HEADS.items():
        w = W[:N].contiguous()
        p4 = w4.quantize(w)
        p8 = w4.quantize8(w)
        print(f"{name}: weight rel error 4-bit {rel(w4.dequantize(p4), w.float()):.4f}, "
              f"8-bit {rel(w4.dequantize8(p8), w.float()):.4f}", flush=True)
        for M in (1, 9):
            x = torch.randn(M, w.shape[1], device="cuda").bfloat16()
            yb = x.float() @ w.float().t()
            for tag, p in (("4-bit", p4), ("8-bit", p8)):
                y = w4.gemm(x, p).float()
                lb, lq = torch.log_softmax(yb, -1), torch.log_softmax(y, -1)
                top = yb.argmax(-1)
                gap = (lb.gather(1, top[:, None]) - lb.gather(1, y.argmax(-1)[:, None])).mean().item()
                kl = (lb.exp() * (lb - lq)).sum(-1).mean().item()
                print(f"   M={M} {tag}: output rel error {rel(y, yb):.4f}, top-1 agree "
                      f"{(y.argmax(-1) == top).float().mean().item():.2f}, KL {kl:.2e}, gap of argmax {gap:.2e}",
                      flush=True)
        del w, p4, p8


def tune(Ms=(1, 9, 16)):
    for name, N in HEADS.items():
        K2 = 2 * 5120
        ps = []
        for _ in range(max(2, (2 << 30) // (N * K2 // 2) + 1)):
            p = make_packed(N, K2, 128)
            p["xdup"] = True
            ps.append(p)
        nb = w4.nbytes(ps[0])
        for M in Ms:
            x = torch.randn(M, 5120, device="cuda").bfloat16()
            out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            res = []
            for cfg in TUNE_CFGS:
                if N % (16 * cfg[0]) or cfg[2] > 1 or w4.lds_bytes(cfg[0], cfg[2], K2, 128) > 65536:
                    continue
                t = graph_time([lambda p=p: w4.gemm(x, p, out=out, cfg=cfg) for p in ps])
                res.append((t, cfg))
            res.sort()
            print(f"{name} {N}x5120 8-bit ({nb / 1e9:.2f} GB) M={M:2d}: " +
                  " ".join(f"{c}:{t * 1e6:.1f}us/{nb / t / 1e12:.2f}TB/s" for t, c in res[:4]), flush=True)
        del ps


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"check": check, "error": error, "tune": tune}[mode]()
