"""Tree / depth / draft-vocab feasibility: verify time at R = K + 1 rows, whole-round time per K, MTP step time at
M rows, GDN verify kernel time, and the draft head (4-bit, M = 1) + argmax per draft-vocabulary size (HIP graphs;
state contents are garbage, lengths are real).
python3 -m eng.bench_verify [K ...]"""
import gc
import os
import sys

import torch

from eng import hip as H
from eng.bench_round import graph, tm
from eng.fast import Fast, Linear, SeqState
from eng.model import Weights

N = int(os.environ.get("BENCH_POS", "10000"))


def main():
    Ks = [int(k) for k in sys.argv[1:]] or [8, 9, 10, 11, 12, 13, 15]
    w = Weights("/models/Qwen3.8-27B")
    st = SeqState(16384)
    x = torch.randn(1, 5120, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(1, dtype=torch.long, device="cuda")
    for V in (32768, 49152, 65536, 98304, 131072, w.lm_head.shape[0]):
        hd = Linear(w.lm_head[:V].contiguous())
        lg = torch.empty(1, V, dtype=torch.bfloat16, device="cuda")
        mp = torch.arange(V, device="cuda")
        g1 = graph(lambda: hd(x, lg))
        g2 = graph(lambda: H.argmax_rows(lg, out, mp))
        print(f"draft head V={V}: gemm {tm(g1.replay, 200) * 1e3:.1f} us, argmax {tm(g2.replay, 200) * 1e3:.1f} us",
              flush=True)
        del hd, g1, g2
    gc.collect()
    torch.cuda.empty_cache()
    for K in Ks:
        F = Fast(w, K)
        F.vt.fill_(100)
        F.set_pos(N)
        F.P[:2] = N
        F.n_acc.fill_(K + 1)
        g = graph(lambda: F.verify(st))
        F.set_pos(N)
        res = [f"verify {tm(g.replay, 50):.3f}"]
        gq = graph(lambda: [H.gdn_verify_q(F.qk[i], F.vv[i], F.gb[i], st.ssm[i], F.ob, F.proj[i],
                                           w.layers[0]["gnorm"], F.goq) for i in range(48)])
        res.append(f"gdn_verify x48 {tm(gq.replay, 50):.3f}")
        F.n_acc.fill_(K + 1)
        ga = graph(lambda: F.accept(st))
        res.append(f"accept+commit {tm(ga.replay, 50):.3f}")
        for M in sorted({1, 2, 3, 4, min(8, K + 1), K + 1}):
            gm = graph(lambda: F.mtp(st, F.new[:M], F.hn[:M], F.P[0:1], F.P[1:2], M, F.zero, F.vt[1:2]))
            F.set_pos(N)
            F.P[:2] = N
            res.append(f"mtp M={M} {tm(gm.replay, 50):.3f}")
        F.capture(st)

        def rnd():
            F.set_pos(N)
            F.g_round.replay()
        res.append(f"whole round {tm(rnd, 50):.3f}")
        print(f"R={K + 1} (K={K}): " + ", ".join(res) + " ms", flush=True)
        del F, g, gq, ga, gm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
