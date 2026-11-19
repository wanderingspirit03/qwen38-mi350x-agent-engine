"""M = 1 GEMV (w4.gemv) vs the w4 gemm kernel on the MTP step's shapes: correctness and time (HIP graphs,
weights rotated over copies so that every call reads from HBM). python3 -m eng.bench_gv [shape ...]"""
import itertools
import sys

import torch

from eng.kernels import w4

SHAPES = {   # name: N, K, rms (norm folded), swiglu, residual, argmax
    "fc": (5120, 10240, False, False, False, False),
    "qkv": (14336, 5120, True, False, False, False),
    "o": (5120, 6144, False, False, True, False),
    "gate_up": (34816, 5120, True, True, False, False),
    "down": (5120, 17408, False, False, True, False),
    "dhead": (33024, 5120, False, False, False, True),
}


def am_take(am):
    k = int(am.max().item())
    am.zero_()
    return 0x7fffffff - (k & 0xffffffff)


def gtime(calls, reps=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls:
            c()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            for c in calls:
                c()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = 1e9
    for _ in range(5):
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1e3 / reps / len(calls))
    return best


def main(names):
    torch.manual_seed(0)
    am = w4.argmax_slots("cuda")
    for name in names or SHAPES:
        N, K, rms, sw, resid, amx = SHAPES[name]
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        nw = torch.randn(K, device="cuda", dtype=torch.bfloat16) * 0.1 if rms else None
        p = w4.quantize(w, norm_weight=nw, swiglu=sw, clip_search=False)
        del w
        x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16) * (3.0 if rms else 1.0)
        No = N // 2 if sw else N
        r0 = torch.randn(1, No, device="cuda", dtype=torch.bfloat16)
        old_cfg = w4.config_for(N, K, 1)

        def run(new, cfg=None, pp=p, out=None, i8=False):
            out = r0.clone() if out is None else out
            kw = dict(residual=out if resid else None, argmax=amx)
            if new:
                w4.gemv(x, pp, out, cfg=cfg, i8=i8, **kw)
            else:
                w4.gemm(x, pp, out=out, cfg=old_cfg, **kw)
            return out
        ya = run(False)
        ia = am_take(am) if amx else -1
        yr = w4.reference(x, p, residual=r0 if resid else None)
        rel = lambda y: (y.float() - yr).abs().max().item() / yr.abs().max().item()
        copies = [p] + [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in p.items()}
                        for _ in range(max(1, 600_000_000 // w4.nbytes(p)))]
        mb = w4.nbytes(p) / 1e6
        outs = [r0.clone() for _ in copies]
        t_old = gtime([lambda q=q, o=o: run(False, pp=q, out=o) for q, o in zip(copies, outs)])
        print(f"{name:8s} N={N} K={K} {mb:.1f} MB  old {old_cfg}: {t_old:6.2f} us ({mb / t_old:.2f} TB/s)  rel err vs fp32 {rel(ya):.2e}", flush=True)
        res = []
        for nt, wv, d, sk in itertools.product((1, 2), (2, 4, 8), (1, 2, 3, 4), (1, 2)):
            i8 = False
            if sw and nt % 2:
                continue
            cfg = (nt, wv, d, sk)
            try:
                yb = run(True, cfg, i8=i8)
            except AssertionError:
                continue
            ib = am_take(am) if amx else -1
            err = rel(yb)
            t = gtime([lambda q=q, o=o: run(True, cfg, pp=q, out=o, i8=i8) for q, o in zip(copies, outs)])
            res.append((t, cfg + (("i8",) if i8 else ()), err, ia == ib))
        res.sort()
        for t, cfg, err, ok in res[:6] + [v for v in res if v[1][3] == 1][:1]:
            print(f"   new {cfg}: {t:6.2f} us ({mb / t:.2f} TB/s)  rel err vs fp32 {err:.2e}"
                  + (f"  argmax {'same' if ok else 'DIFFERENT'}" if amx else ""), flush=True)
        bad = [(cfg, err, ok) for t, cfg, err, ok in res if err > 2e-2 or not ok]
        if bad:
            print("   MISMATCH:", bad, flush=True)
        del copies, outs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:])
