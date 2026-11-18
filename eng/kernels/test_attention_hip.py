"""Test + benchmark of the HIP decode attention (eng.kernels.attention_hip) against the fp32 reference, with the
Triton attn_decode for comparison. Run: ./try tests/in_container.sh python3 -m eng.kernels.test_attention_hip [sweep]"""
import sys

import torch

from .attention import attn_decode, HEADS, HEAD_DIM
from .attention_hip import attn_decode_hip, attn_decode_best
from .test_attention import ref_attn, make_cache, make_q, bench, bench_graph, check, dev


def main():
    torch.manual_seed(0)
    ok = True
    print("== correctness: HIP decode", flush=True)
    for T, n, sc in [(1, 1, 1.0), (1, 17, 1.0), (4, 4, 1.0), (8, 100, 1.0), (16, 3000, 1.0), (8, 12000, 3.0),
                     (16, 16000, 1.0), (3, 65, 1.0), (12, 777, 2.0), (2, 5000, 1.0), (11, 4097, 1.0), (16, 16, 1.0)]:
        k, v = make_cache(n, sc)
        q, gate = make_q(T, sc)
        ref = ref_attn(q, k, v, n)
        nt = torch.tensor([n], dtype=torch.int32, device=dev)
        ok &= check(f"hip T={T} n={n} qk-scale={sc}", attn_decode_hip(q, k, v, nt), ref)
        if T in (8, 16):
            ok &= check("  fp32 partials", attn_decode_hip(q, k, v, nt, part_bf16=False), ref)
            ok &= check("  gated", attn_decode_hip(q, k, v, nt, gate=gate), ref * torch.sigmoid(gate.float()))
            pt = torch.tensor([n - T], dtype=torch.int32, device=dev)
            ok &= check("  start position + n_add", attn_decode_hip(q, k, v, pt, n_add=T), ref)
    k, v = make_cache(16000)
    q, _ = make_q(8)
    nt = torch.tensor([100], dtype=torch.int32, device=dev)
    out = torch.empty(8, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    attn_decode_hip(q, k, v, nt, out=out)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        attn_decode_hip(q, k, v, nt, out=out)
    for n in [100, 5000, 16000, 9]:
        nt.fill_(n)
        gr.replay()
        ok &= check(f"hip in graph, replay n={n}", out, ref_attn(q, k, v, n))
    for T in (4, 8):
        k, v = make_cache(3000)
        q, gate = make_q(T)
        nt = torch.tensor([3000], dtype=torch.int32, device=dev)
        ok &= check(f"attn_decode_best T={T} gated", attn_decode_best(q, k, v, nt, gate=gate),
                    ref_attn(q, k, v, 3000) * torch.sigmoid(gate.float()))
    print("ALL OK" if ok else "SOME FAILED", flush=True)

    print("== decode timing per layer (us, CUDA-graph replay): HIP / Triton", flush=True)
    k, v = make_cache(16000)
    for T in (1, 2, 4, 8, 12, 16):
        q, _ = make_q(T)
        line = []
        for n in (500, 4000, 12000, 16000):
            nt = torch.tensor([n], dtype=torch.int32, device=dev)
            th = bench_graph(lambda: attn_decode_hip(q, k, v, nt))
            tt = bench_graph(lambda: attn_decode(q, k, v, nt))
            line.append(f"n={n}: {th:5.1f} / {tt:5.1f}")
        print(f"   T={T:2d}  " + "  ".join(line), flush=True)
    if "sweep" in sys.argv:
        for T in (1, 4, 8, 12, 16):
            q, _ = make_q(T)
            for n in (500, 12000):
                nt = torch.tensor([n], dtype=torch.int32, device=dev)
                for ns in ((64,) if n == 12000 else (64,)):
                    us = [bench_graph(lambda: attn_decode_hip(q, k, v, nt, nsplit=ns, stages=st))
                          for st in (3, 1, 5, 2, 9, 17)]
                    print(f"   T={T:2d} n={n:5d} nsplit={ns:3d}: {us[0]:5.1f} us (split {us[1]:5.1f}, split without"
                          f" math {us[2]:5.1f}, merge {us[3]:5.1f}; split without QK {us[4]:5.1f}, without PV"
                          f" {us[5]:5.1f})", flush=True)
    print("   (KV bytes at n=12000: 49.2 MB = 10.5 us at 4.7 TB/s)", flush=True)


if __name__ == "__main__":
    main()
