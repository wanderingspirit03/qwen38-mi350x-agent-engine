"""Test + benchmark for eng.kernels.w4.
python3 -m eng.kernels.test_w4 MODE   with MODE one of:
  allcfg   every kernel instance and every fusion against w4.reference (random weights)
  check    quantization error and kernel error on real weights of all shapes
  tune     time every configuration per shape and M (inside HIP graphs, weights not in cache)
  bench    time the configured (default) path per shape and M
"""
import json
import os
import sys
import time

import torch
from safetensors import safe_open

from . import w4

MODEL = "/models/Qwen3.8-27B"
P = "model.language_model.layers."
SHAPES = {  # name: (N, K, tensors concatenated along N, norm weight)
    "gate_up": (34816, 5120, [P + "0.mlp.gate_proj.weight", P + "0.mlp.up_proj.weight"], P + "0.post_attention_layernorm.weight"),
    "down": (5120, 17408, [P + "0.mlp.down_proj.weight"], None),
    "gdn_in": (16480, 5120, [P + "0.linear_attn.in_proj_qkv.weight", P + "0.linear_attn.in_proj_z.weight",
                             P + "0.linear_attn.in_proj_b.weight", P + "0.linear_attn.in_proj_a.weight"],
               P + "0.input_layernorm.weight"),
    "out": (5120, 6144, [P + "0.linear_attn.out_proj.weight"], None),
    "qkv": (14336, 5120, [P + "3.self_attn.q_proj.weight", P + "3.self_attn.k_proj.weight",
                          P + "3.self_attn.v_proj.weight"], P + "3.input_layernorm.weight"),
    "lm_head": (248320, 5120, ["lm_head.weight"], "model.language_model.norm.weight"),
}
MS = (1, 2, 4, 8, 16)


def load(names):
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    out = []
    for n in names:
        with safe_open(os.path.join(MODEL, idx[n]), framework="pt", device="cpu") as f:
            out.append(f.get_tensor(n))
    return torch.cat(out).cuda()


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def check_all_configs(N=5120, K=6144):
    """Every kernel instance and fusion against w4.reference on random weights."""
    w = torch.randn(N, K, device="cuda").bfloat16()
    nw = (torch.randn(K, device="cuda") * 0.3).bfloat16()
    bad = n = 0
    for G in (128, 64):
        for rms, swiglu in ((False, False), (True, False), (True, True), (False, True)):
            p = w4.quantize(w, group=G, norm_weight=nw if rms else None, swiglu=swiglu)
            for M in (1, 3, 8, 16):
                x = (torch.randn(M, K, device="cuda") * 3).bfloat16()
                r = torch.randn(M, N, device="cuda").bfloat16()
                yr = w4.reference(x, p)
                yrr = w4.reference(x, p, residual=r)
                for nt in (1, 2, 4, 8):
                    for waves in (1, 2, 4, 8, 16):
                        for ks in (1, 3):
                            for ntl in (1,):
                                cfg = (nt, waves, ks, ntl)
                                if nt * -(-K // 128 // ks) * (128 // G) * 48 > 65536:
                                    continue
                                y = w4.gemm(x, p, cfg=cfg)
                                e = rel(y, yr)
                                e2 = 0
                                if not swiglu:   # in-place residual
                                    rr = r.clone()
                                    w4.gemm(x, p, out=rr, residual=rr, cfg=cfg)
                                    e2 = rel(rr, yrr)
                                n += 1
                                if not (e < 5e-3 and e2 < 5e-3):
                                    bad += 1
                                    print(f"FAIL G={G} rms={rms} swiglu={swiglu} M={M} cfg={cfg} err {e:.3e} {e2:.3e}",
                                          flush=True)
    print(f"all configs: {bad} failures of {n}", flush=True)
    # the fused path against the plain PyTorch ops of the model, with the unquantized weights as reference:
    # the fused error should be about the plain 4-bit error
    from .. import ref_ops as R
    for M in (1, 16):
        x = (torch.randn(M, K, device="cuda") * 3).bfloat16()
        y0 = R.rmsnorm(x, nw).float() @ w.float().t()
        y1 = w4.gemm(x, w4.quantize(w, norm_weight=nw))
        y2 = w4.gemm(R.rmsnorm(x, nw), w4.quantize(w))
        print(f"vs bf16 rmsnorm + matmul, M={M}: fused norm {rel(y1, y0):.4f}, separate norm {rel(y2, y0):.4f}",
              flush=True)


def check(names, groups=(128, 64)):
    for name in names:
        N, K, tn, nn = SHAPES[name]
        w = load(tn)
        nw = load([nn]) if nn else None
        assert w.shape == (N, K), (name, w.shape)
        for G in groups:
            torch.cuda.synchronize()
            t = time.perf_counter()
            p = w4.quantize(w, group=G)
            torch.cuda.synchronize()
            tq = time.perf_counter() - t
            wd = w4.dequantize(p)
            e = rel(wd, w)
            x = torch.randn(16, K, device="cuda").bfloat16()
            yb = x.float() @ w.float().t()
            ek = rel(w4.gemm(x, p), x.float() @ wd.t())
            line = (f"{name:8s} {N}x{K} G={G}: quantize {tq:.2f} s, |W-Wq|/|W| {e:.4f}, output vs bf16 "
                    f"{rel(x.float() @ wd.t(), yb):.4f}, kernel vs dequant {ek:.1e}, {w4.nbytes(p) / 2**20:.1f} MiB")
            if nw is not None:   # error with the norm weight folded in, on normalized activations
                pn = w4.quantize(w, group=G, norm_weight=nw)
                from .. import ref_ops as R
                xn = R.rmsnorm(x, nw)
                line += f"; norm folded: output vs bf16 {rel(w4.gemm(x, pn), xn.float() @ w.float().t()):.4f}"
            print(line, flush=True)
            del p, wd
        del w
        torch.cuda.empty_cache()


def graph_time(calls, reps=5):
    """Capture calls (list of no-arg fns) into one HIP graph, return the mean time per call."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls:
            c()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in calls:
            c()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e-3 / reps / len(calls)


def make_packed(N, K, G, rms=False, swiglu=False):
    return {"qw": torch.randint(-2**31, 2**31 - 1, (N // 16, K // 128, 256), device="cuda", dtype=torch.int32),
            "meta": torch.cat([(torch.rand(N // 16, K // G, 16, device="cuda") * 0.01).half().view(torch.uint8),
                               torch.randint(128, 144, (N // 16, K // G, 16), device="cuda", dtype=torch.uint8)], -1),
            "N": N, "K": K, "G": G, "rms": rms, "swiglu": swiglu}


def bench(names, cfgs=None, Ms=MS, G=128, fused=False, matmul=False, verbose=False):
    """Times each configuration inside a HIP graph, cycling through > 2 GB of weight copies so that nothing
    stays in the 256 MB last-level cache (as in a real forward pass). cfgs None: the configured path.
    fused: with the fusions the engine would use (norm for gate_up/gdn_in/qkv/lm_head, swiglu for gate_up,
    in-place residual for down/out)."""
    w4._workspace("cuda")
    res = {}
    for name in names:
        N, K, _, nn = SHAPES[name]
        rms = fused and nn is not None
        swiglu = fused and name == "gate_up"
        resid = fused and name in ("down", "out")
        ncopy = max(2, (2 << 30) // (N * K // 2) + 1)
        ps = [make_packed(N, K, G, rms, swiglu) for _ in range(ncopy)]
        nb = w4.nbytes(ps[0])
        wb = torch.randn(N, K, device="cuda").bfloat16() if (matmul and name != "lm_head") else None
        for M in Ms:
            x = torch.randn(M, K, device="cuda").bfloat16()
            out = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
            line = f"{name:8s} {N}x{K} M={M:2d}:"
            best = None
            for cfg in (cfgs or [None]):
                if cfg is not None and (N % (16 * cfg[0]) or (cfg[2] > 1 and cfg[2] * (M * N + N // cfg[0] * 16) > w4.WS_FLOATS)
                                        or cfg[0] * -(-K // 128 // cfg[2]) * (128 // G) * 48 > 65536):
                    continue
                t = graph_time([lambda p=p: w4.gemm(x, p, out=out, residual=out if resid else None, cfg=cfg)
                                for p in ps])
                res[(name, M, cfg)] = t
                if best is None or t < best[0]:
                    best = (t, cfg)
                if verbose:
                    line += f" {cfg}: {t * 1e6:6.1f}us {nb / t / 1e12:4.2f} |"
            line += f" {'BEST ' + str(best[1]) if cfgs else ''} {best[0] * 1e6:.1f} us {nb / best[0] / 1e12:.2f} TB/s"
            if wb is not None and M in (1, 16):
                tb = graph_time([lambda: torch.matmul(x, wb.t())] * 20)
                line += f" | bf16 matmul (cached weights) {tb * 1e6:.1f} us"
            print(line, flush=True)
        del ps, wb
        torch.cuda.empty_cache()
    return res


TUNE_CFGS = [(nt, w, ks, 1) for nt in (1, 2, 4, 8) for w in (1, 2, 4, 8, 16) for ks in (1, 2, 4)
             if not (nt == 8 and w == 16)]


def tune(names):
    res = bench(names, TUNE_CFGS)
    for name in names:
        for M in MS:
            r = sorted((t, c) for (n, m, c), t in res.items() if n == name and m == M)[:4]
            print(name, M, " ".join(f"{c}:{t * 1e6:.1f}" for t, c in r), flush=True)
    return res


VARIANTS = {"base": None, "x_once": "#define X_ONCE\n"}


def variants(names=("gate_up", "down", "out", "lm_head")):
    """What the per-group scale/zero loads and the epilogue math cost (the cut variants give wrong results)."""
    cfgs = [(2, 8, 1, 1), (2, 16, 1, 1), (4, 2, 1, 1), (2, 1, 1, 1)]
    for v, pre in VARIANTS.items():
        w4._lib = None
        w4._lib = w4._build(pre) if pre else w4._build()
        print("variant", v, flush=True)
        bench(list(names), cfgs, Ms=(1, 16))
    w4._lib = None


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "allcfg"
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(SHAPES)
    pr = torch.cuda.get_device_properties(0)
    print(pr.name, "CUs", pr.multi_processor_count, flush=True)
    t = time.time()
    w4._build()
    print(f"build {time.time() - t:.1f} s", flush=True)
    if mode == "allcfg":
        check_all_configs()
    if mode == "check":
        check(names)
    if mode == "tune":
        tune(names)
    if mode == "variants":
        variants()
    if mode == "bench":
        bench(names)
        bench(names, fused=True, Ms=(1, 16))


if __name__ == "__main__":
    main()
