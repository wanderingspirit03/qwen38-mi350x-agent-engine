"""Ablation of the M=1 GEMV on gate_up / down shapes (random weights): full vs no math vs loads only, and the
plain read of the same bytes. python3 -m eng.bench_gv_abl"""
import torch

from eng.bench_gv import gtime
from eng.kernels import w4

torch.manual_seed(0)
lib = w4._build()
rdout = torch.zeros(4, dtype=torch.int32, device="cuda")
for name, N, K, rms, sw, cfgs in (("gate_up", 34816, 5120, True, True, [(2, 4, 2), (2, 8, 1), (2, 8, 2)]),
                                  ("down", 5120, 17408, False, False, [(2, 8, 2), (2, 4, 2)]),
                                  ("qkv", 14336, 5120, False, False, [(2, 4, 1), (2, 4, 2)])):
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    nw = torch.randn(K, device="cuda", dtype=torch.bfloat16) * 0.1 if rms else None
    p = w4.quantize(w, norm_weight=nw, swiglu=sw, clip_search=False)
    del w
    x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    No = N // 2 if sw else N
    copies = [p] + [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in p.items()}
                    for _ in range(max(1, 600_000_000 // w4.nbytes(p)))]
    outs = [torch.zeros(1, No, device="cuda", dtype=torch.bfloat16) for _ in copies]
    mb = w4.nbytes(p) / 1e6
    st = torch.cuda.current_stream().cuda_stream
    t = gtime([lambda q=q: lib.rd_bench(q["qw"].data_ptr(), q["qw"].numel() * 4, rdout.data_ptr(), 1024, 4, st)
               for q in copies])
    print(f"{name} {mb:.1f} MB: plain read of qw {t:.2f} us ({mb / t:.2f} TB/s)", flush=True)
    for cfg in cfgs:
        if rms != p["rms"]:
            continue
        r = []
        for mode in (0, 1, 2):
            if mode == 0:
                calls = [lambda q=q, o=o: w4.gemv(x, q, o, cfg=cfg) for q, o in zip(copies, outs)]
            else:
                calls = [lambda q=q, o=o, m=mode: w4.gemv_mode(x, q, o, cfg, m) for q, o in zip(copies, outs)]
            try:
                tt = gtime(calls)
            except AssertionError as e:
                r.append(f"mode {mode}: n/a")
                continue
            r.append(f"{['full', 'no math', 'loads only'][mode]} {tt:.2f} us ({mb / tt:.2f} TB/s)")
        print(f"   {cfg}: " + " | ".join(r), flush=True)
    del copies, outs
    torch.cuda.empty_cache()
