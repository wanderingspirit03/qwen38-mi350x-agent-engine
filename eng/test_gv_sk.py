import torch, traceback
from eng.kernels import w4
from eng.bench_gv import gtime
torch.manual_seed(0)
for N, K in ((5120, 17408), (5120, 6144), (5120, 10240), (14336, 5120)):
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    p = w4.quantize(w, clip_search=False)
    x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    ref = (x.float() @ w.float().t())
    copies = [p] + [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in p.items()} for _ in range(max(1, 600_000_000 // w4.nbytes(p)))]
    outs = [torch.zeros(1, N, device="cuda", dtype=torch.bfloat16) for _ in copies]
    for cfg in ((2, 8, 2, 1), (2, 4, 2, 2), (1, 4, 2, 2), (2, 4, 1, 2), (2, 2, 2, 2), (1, 2, 2, 4), (2, 4, 2, 4), (1, 4, 2, 4), (2, 8, 2, 2), (1, 8, 2, 2)):
        try:
            y = w4.gemv(x, p, torch.zeros(1, N, device="cuda", dtype=torch.bfloat16), cfg=cfg)
            torch.cuda.synchronize()
            y2 = w4.gemv(x, p, torch.zeros(1, N, device="cuda", dtype=torch.bfloat16), cfg=cfg)
            if cfg[3] == 1 and cfg[:3] == (2, 8, 2):
                y_base = y.clone()
            err = (y.float() - y_base.float()).abs().max().item()
            err2 = (y2.float() - y.float()).abs().max().item()
            t = gtime([lambda q=q, o=o: w4.gemv(x, q, o, cfg=cfg) for q, o in zip(copies, outs)])
            err3 = max((o.float() - y.float()).abs().max().item() for o in outs)
            print(N, K, cfg, f"{t:.2f} us  max diff vs sk=1 {err:.1e}, rerun {err2:.1e}, after graph {err3:.1e}", flush=True)
        except Exception as e:
            print(N, K, cfg, "FAIL", repr(e)[:200], flush=True)
    del copies, outs
    torch.cuda.empty_cache()
