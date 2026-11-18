"""Test + benchmark for eng.kernels.w4a8 (W4A8 decode GEMM).
python3 -m eng.kernels.test_w4a8 MODE [shapes] [Ms]
  allcfg   every kernel instance and fusion against w4a8.reference (random weights)
  tune     time configurations per shape and M (HIP graphs, weights not in cache), w4 tuned config for comparison
  bench    configured path: w4a8 GEMM alone, quant_x + GEMM, and w4
"""
import sys
import time

import torch

from . import w4, w4a8
from .test_w4 import SHAPES, graph_time, rel, make_packed as make_packed4


def check_all_configs(N=5120, K=6144):
    w = torch.randn(N, K, device="cuda").bfloat16()
    nw = (torch.randn(K, device="cuda") * 0.3).bfloat16()
    bad = n = 0
    for rms, swiglu in ((False, False), (True, False), (True, True), (False, True)):
        p = w4a8.quantize(w, norm_weight=nw if rms else None, swiglu=swiglu)
        for M in (1, 3, 9, 16):
            x = (torch.randn(M, K, device="cuda") * 3).bfloat16()
            r = torch.randn(M, N, device="cuda").bfloat16()
            yr = w4a8.reference(x, p)
            yrr = w4a8.reference(x, p, residual=r)
            for nt in (1, 2, 4, 8):
                for waves in (1, 2, 4, 8):
                    if swiglu and nt % 2:
                        continue
                    y = w4a8.gemm(x, p, cfg=(nt, waves))
                    e = rel(y, yr)
                    e2 = 0
                    if not swiglu:
                        rr = r.clone()
                        w4a8.gemm(x, p, out=rr, residual=rr, cfg=(nt, waves))
                        e2 = rel(rr, yrr)
                    n += 1
                    if not (e < 5e-3 and e2 < 5e-3):
                        bad += 1
                        print(f"FAIL rms={rms} swiglu={swiglu} M={M} cfg={(nt, waves)} err {e:.3e} {e2:.3e}", flush=True)
    print(f"all configs: {bad} failures of {n}", flush=True)
    # against unquantized bf16 math (x normed by the model's rmsnorm)
    from .. import ref_ops as R
    for M in (1, 16):
        x = (torch.randn(M, K, device="cuda") * 3).bfloat16()
        y0 = R.rmsnorm(x, nw).float() @ w.float().t()
        y1 = w4a8.gemm(x, w4a8.quantize(w, norm_weight=nw))
        y4 = w4.gemm(R.rmsnorm(x, nw), w4.quantize(w))
        print(f"vs bf16 rmsnorm + matmul, M={M}: w4a8 fused norm {rel(y1, y0):.4f}, w4 {rel(y4, y0):.4f}", flush=True)


def check_gdnq():
    """eng.hip.gdn_verify_q against gdn_verify + quant_x."""
    from .. import hip as H
    from ..fast import PROJ
    T = 9
    qk = torch.randn(T, 32, 128, device="cuda") * 0.1
    v = torch.randn(T, 48, 128, device="cuda")
    gb = torch.stack([-torch.rand(T, 48, device="cuda"), torch.rand(T, 48, device="cuda")], 1)
    st = torch.randn(48, 128, 128, device="cuda") * 0.1
    o = torch.empty(T, 48, 128, device="cuda")
    proj = torch.randn(T, PROJ, device="cuda").bfloat16()
    gw = torch.rand(128, device="cuda").bfloat16()
    out = torch.empty(T, 6144, device="cuda").bfloat16()
    H.gdn_verify(qk, v, gb, st, o, proj, gw, out)
    ref = w4a8.quant_x(out)
    xa = w4a8.new_xa(T, 6144)
    H.gdn_verify_q(qk, v, gb, st, o, proj, gw, xa)
    e = [(xa[k].float() - ref[k].float()).abs().max().item() for k in ("xq", "xzp")]
    e.append((xa["xg"][..., 0] - ref["xg"][..., 0]).abs().max().item() / ref["xg"][..., 0].abs().max().item())
    print(f"gdn_verify_q vs gdn_verify + quant_x: max diffs xq {e[0]}, xzp {e[1]:.3g}, rel sx {e[2]:.3g}", flush=True)


def check_attnq():
    """attn_decode(xa=...) against attn_decode + quant_x."""
    from .attention import attn_decode
    for T in (1, 9):
        q = torch.randn(T, 24, 256, device="cuda").bfloat16()
        kc = torch.randn(4096, 4, 256, device="cuda").bfloat16()
        vc = torch.randn(4096, 4, 256, device="cuda").bfloat16()
        gate = torch.randn(T, 24, 256, device="cuda").bfloat16()
        n = torch.tensor([3000], dtype=torch.int32, device="cuda")
        xa = w4a8.new_xa(T, 6144)
        o = attn_decode(q, kc, vc, n, gate=gate, xa=xa)
        ref = w4a8.quant_x(o.reshape(T, 6144))
        e = [(xa[k].float() - ref[k].float()).abs().max().item() for k in ("xq", "xzp")]
        e.append((xa["xg"][..., 0] - ref["xg"][..., 0]).abs().max().item() / ref["xg"][..., 0].abs().max().item())
        print(f"attn_decode xa vs quant_x, T={T}: max diffs xq {e[0]}, xzp {e[1]:.3g}, rel xg {e[2]:.3g}", flush=True)


def make_packed(N, K, rms=False, swiglu=False):
    NG = K // 128
    NGp = -(-NG // 32) * 32
    return {"qw": torch.randint(-2**31, 2**31 - 1, (N // 16, NG, 256), device="cuda", dtype=torch.int32),
            "msc": (torch.rand(N // 16, NG, 16, device="cuda") * 0.01).half(),
            "mzp": (torch.rand(N // 16, 16, NGp, device="cuda") * 0.01).bfloat16(),
            "N": N, "K": K, "rms": rms, "swiglu": swiglu,
            "norm": torch.zeros(K, device="cuda", dtype=torch.bfloat16) if rms else None}


CFGS = [(nt, w) for nt in (1, 2, 4, 8) for w in (1, 2, 4, 8)]


def bench(names, Ms, cfgs=None, fused=True, compare=True):
    res = {}
    for name in names:
        N, K, _, nn = SHAPES[name]
        rms = fused and nn is not None
        swiglu = fused and name == "gate_up"
        resid = fused and name in ("down", "out")
        ncopy = max(2, (2 << 30) // (N * K // 2) + 1)
        ps = [make_packed(N, K, rms, swiglu) for _ in range(ncopy)]
        nb = w4a8.nbytes(ps[0])
        p4 = [make_packed4(N, K, 128, rms, swiglu) for _ in range(ncopy)] if compare else None
        for M in Ms:
            x = torch.randn(M, K, device="cuda").bfloat16()
            xa = w4a8.quant_x(x)
            out = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
            line = f"{name:8s} {N}x{K} M={M:2d}:"
            best = None
            for cfg in (cfgs or [None]):
                c = cfg or w4a8.config_for(N, K, M)
                if N % (16 * c[0]) or (swiglu and c[0] % 2) or c[0] * (K // 128) * 32 > 65536:
                    continue
                t = graph_time([lambda p=p: w4a8.gemm(xa, p, out=out, residual=out if resid else None, cfg=cfg)
                                for p in ps])
                res[(name, M, cfg)] = t
                if cfgs:
                    line += f" {c[0]},{c[1]}:{t * 1e6:.1f}"
                if best is None or t < best[0]:
                    best = (t, c)
            line += f" w4a8 {best[1]} {best[0] * 1e6:.1f} us {nb / best[0] / 1e12:.2f} TB/s"
            if cfgs is None:
                tq = graph_time([lambda p=p: w4a8.gemm(x, p, out=out, xa=xa, residual=out if resid else None)
                                 for p in ps])
                line += f" | +quant_x {tq * 1e6:.1f} us"
            if compare:
                t4 = graph_time([lambda p=p: w4.gemm(x, p, out=out, residual=out if resid else None) for p in p4])
                line += f" | w4 {w4.config_for(N, K, M)} {t4 * 1e6:.1f} us {w4.nbytes(p4[0]) / t4 / 1e12:.2f} TB/s"
            print(line, flush=True)
        del ps, p4
        torch.cuda.empty_cache()
    return res


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "allcfg"
    names = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else [
        "gate_up", "down", "gdn_in", "out", "qkv", "mtp_fc", "dhead"]
    Ms = tuple(int(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else (1, 9)
    t = time.time()
    w4a8._build()
    w4._build()
    w4._workspace("cuda")
    print(f"build {time.time() - t:.1f} s", flush=True)
    if mode == "allcfg":
        check_all_configs()
    if mode == "gdnq":
        check_gdnq()
        check_attnq()
    if mode == "tune":
        bench(names, Ms, CFGS)
    if mode == "bench":
        bench(names, Ms)


if __name__ == "__main__":
    main()
