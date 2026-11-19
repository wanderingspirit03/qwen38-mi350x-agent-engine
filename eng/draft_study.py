"""Draft acceptance study: teacher-forced MTP chains, batched over anchors in fp32 PyTorch, for variants of the MTP
layer weights (w4 with the norm folded = the engine, w4 unfolded, w8, bf16), of the draft head (4-bit or bf16,
draft vocabulary + prompt extra or full vocabulary), on two paths:
  E: the engine's greedy output (truth = engine tokens), target hidden from the fp8 prefill path (as tree_sim)
  B: the bf16 target's greedy output (stepwise bf16 prefill, 8-bit attention off), target hidden from it.
Anchor p (round start token S[p], p >= n): depth j drafts S[p+j]; depth 1 = catch-up row (hidden at p-1, token
S[p], position p-1), depth j >= 2 = chain row (MTP hidden of depth j-1, token S[p+j-1], position p+j-2) seeing the
real MTP keys of positions < p and its own chain keys. Output: 'DSP' pieces / 'DSE' (json, match bits per variant).
python3 -m eng.draft_study PROMPTS.jsonl [N] [MAXT] [D] [FIRST]
"""
import base64
import json
import os
import sys
import time

import numpy as np
import torch

from . import prefill as PF
from .engine import Engine
from .kernels import w4 as w4q
from .main import MODEL
from .model import HEAD_DIM, HEADS, HIDDEN, INTER, KV_HEADS, apply_rope, rope_cos_sin

torch.backends.cuda.matmul.allow_tf32 = False
EPS = 1e-6


def rms(x, w=None):
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return x if w is None else x * (1.0 + w.float())


def fq4(w, nw=None, G=128):
    """fp32 (N, K): w4.quantize's codes, dequantized (norm weight folded in if given)."""
    out = torch.empty(w.shape, dtype=torch.float32, device=w.device)
    for r0 in range(0, w.shape[0], 8192):
        wc = w[r0:r0 + 8192].float()
        if nw is not None:
            wc = wc * (1 + nw.float())[None]
        u, s, z = w4q.rtn_search(wc.reshape(wc.shape[0], -1, G))
        out[r0:r0 + 8192] = ((u.float() - z[..., None]) * s[..., None]).reshape(wc.shape)
    return out


def fq8(w, nw=None):
    """fp32 (N, K): w4.quantize8's values (symmetric int8 per group of 128)."""
    wc = w.float()
    if nw is not None:
        wc = wc * (1 + nw.float())[None]
    wc = wc.reshape(w.shape[0], -1, 128)
    s = (wc.abs().amax(-1) / 127).clamp(min=1e-8).half().float()
    return ((wc / s[..., None]).round().clamp_(-127, 127) * s[..., None]).reshape(w.shape)


class MtpW:
    """MTP layer weights of one variant: fp32 matrices; fold=True: the in/post norm weights are inside W.
    kind: w4 (the engine), w4nf (norm not folded), w4g64 (groups of 64), w8, bf, or w4+<matrix>8 (that matrix
    in 8 bits: fc, qkv, o, gate_up, down)."""

    def __init__(self, M, kind):
        self.M, self.kind = M, kind
        self.fold = kind not in ("bf", "w4nf")
        base, _, hi = kind.partition("+")
        g64 = lambda w, nw=None: fq4(w, nw, 64)
        q = {"w4": fq4, "w4nf": fq4, "w4g64": g64, "w8": fq8, "bf": lambda w, nw=None: w.float(),
             "gq": lambda w, nw=None: w.float()}[base]
        norms = {"qkv": "in_norm", "gate_up": "post_norm"}
        for k in ("fc", "qkv", "o", "gate_up", "down"):
            qk = fq8 if hi == k + "8" else q
            nw = M[norms[k]] if self.fold and k in norms else None
            setattr(self, k, qk(M[k], nw))

    def norm_in(self, x, name):
        return rms(x, None if self.fold else self.M[name])


def gptq(W, H, G=128, damp=0.01):
    """GPTQ (no act-order) into w4.quantize's format: asymmetric 4-bit per group of G along K, scale and zero
    from rtn_search on the error-updated group. W (N, K) fp32 (norm already folded), H = X^T X (K, K).
    Returns the dequantized fp32 matrix."""
    W = W.float().clone()
    N, K = W.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += torch.eye(K, device=H.device) * damp * torch.diag(H).mean()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, K, G):
        i2 = i1 + G
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        _, s, z = w4q.rtn_search(W1[:, None, :])
        s, z = s[:, 0], z[:, 0]
        for i in range(G):
            w = W1[:, i]
            q = ((w / s).round() + z).clamp(0, 15)
            q = (q - z) * s
            Q[:, i1 + i] = q
            e = (w - q) / Hi[i, i]
            W1[:, i:] -= e[:, None] * Hi[i, i:][None]
            E1[:, i] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


@torch.inference_mode()
def calib(W, emb, S, HN, Hs):
    """Accumulate X^T X of every MTP linear's input (bf16-weight MTP W, all rows of S) and of the MTP output
    (draft head input) into Hs."""
    L = S.shape[0]
    K1 = torch.empty(L - 1, KV_HEADS, HEAD_DIM, device=S.device)
    V1 = torch.empty_like(K1)
    xs = []
    for r0 in range(0, L - 1, 2048):
        r1 = min(L - 1, r0 + 2048)
        pos = torch.arange(r0, r1, device=S.device)
        M = W.M
        xin = torch.cat([rms(emb[S[r0 + 1:r1 + 1]], M["pre_emb"]), rms(HN[r0:r1], M["pre_hid"])], -1)
        x = xin @ W.fc.t()
        q, gate, k, v = qkv_rows(W, x, pos)
        K1[r0:r1], V1[r0:r1] = k, v
        xs.append((pos, xin, x, q, gate))

    def acc(name, X):
        X = X.float()
        Hs[name] = Hs.get(name, 0) + X.t() @ X

    for pos, xin, x, q, gate in xs:
        acc("fc", xin)
        acc("qkv", rms(x))
        a = attention(q, K1, V1, pos + 1) * torch.sigmoid(gate)
        acc("o", a)
        x = x + a @ W.o.t()
        acc("gate_up", rms(x))
        gu = rms(x, W.M["post_norm"]) @ W.gate_up.t()
        act = torch.nn.functional.silu(gu[:, :INTER]) * gu[:, INTER:]
        acc("down", act)
        x = x + act @ W.down.t()
        acc("head", rms(x, W.M["norm"]))


def attention(q, K1, V1, lim, Kc=None, Vc=None):
    """q (R, 24, 256) fp32; real keys K1/V1 (P, 4, 256), row r sees positions < lim[r]; extra keys Kc/Vc
    (R, m, 4, 256) all visible to their row. Returns (R, 24 * 256)."""
    R, G = q.shape[0], HEADS // KV_HEADS
    qg = q.view(R, KV_HEADS, G, HEAD_DIM) * HEAD_DIM ** -0.5
    hi = int(lim.max())
    s = torch.einsum("rkgd,pkd->rkgp", qg, K1[:hi])
    s.masked_fill_((torch.arange(hi, device=q.device)[None] >= lim[:, None])[:, None, None], float("-inf"))
    if Kc is not None:
        s = torch.cat([s, torch.einsum("rkgd,rmkd->rkgm", qg, Kc)], -1)
    p = s.softmax(-1)
    o = torch.einsum("rkgp,pkd->rkgd", p[..., :hi], V1[:hi])
    if Kc is not None:
        o = o + torch.einsum("rkgm,rmkd->rkgd", p[..., hi:], Vc)
    return o.reshape(R, HEADS * HEAD_DIM)


def qkv_rows(W, x, pos):
    """x (R, H) fp32 residual after fc -> q (R, 24, 256), gate (R, 24*256), k, v (R, 4, 256), rope at pos."""
    M, R = W.M, x.shape[0]
    qkv = W.norm_in(x, "in_norm") @ W.qkv.t()
    qg = qkv[:, :HEADS * 512].view(R, HEADS, 512)
    q, gate = qg[..., :HEAD_DIM], qg[..., HEAD_DIM:].reshape(R, -1)
    k = qkv[:, HEADS * 512:HEADS * 512 + KV_HEADS * HEAD_DIM].view(R, KV_HEADS, HEAD_DIM)
    v = qkv[:, HEADS * 512 + KV_HEADS * HEAD_DIM:].view(R, KV_HEADS, HEAD_DIM)
    cos, sin = rope_cos_sin(pos, x.device)
    q = apply_rope(rms(q, M["q_norm"]), cos, sin)
    k = apply_rope(rms(k, M["k_norm"]), cos, sin)
    return q, gate, k, v


def fc_rows(W, emb, tok, hid):
    M = W.M
    return torch.cat([rms(emb[tok], M["pre_emb"]), rms(hid, M["pre_hid"])], -1) @ W.fc.t()


def finish(W, x, q, gate, K1, V1, lim, Kc=None, Vc=None):
    """attention + o + MLP + MTP norm -> normed hidden (R, H) fp32"""
    a = attention(q, K1, V1, lim, Kc, Vc) * torch.sigmoid(gate)
    x = x + a @ W.o.t()
    gu = W.norm_in(x, "post_norm") @ W.gate_up.t()
    x = x + (torch.nn.functional.silu(gu[:, :INTER]) * gu[:, INTER:]) @ W.down.t()
    return rms(x, W.M["norm"])


@torch.inference_mode()
def chains(W, emb, S, n, HN, D, Kctx, Vctx):
    """S (L,) long on the device, HN (L, H) target hidden, Kctx/Vctx: the prefill path's MTP keys of positions
    0..n-1 (as the engine has them). Returns per depth the MTP normed hidden of every anchor p = n..L-2:
    list of (A_j, H) fp32 (anchors with p + j <= L - 1)."""
    L = S.shape[0]
    # real MTP keys of positions 0..L-2 (row i: hidden i, token i+1)
    K1 = torch.empty(L - 1, KV_HEADS, HEAD_DIM, device=S.device)
    V1 = torch.empty_like(K1)
    keep = {}
    for r0 in range(0, L - 1, 2048):
        r1 = min(L - 1, r0 + 2048)
        pos = torch.arange(r0, r1, device=S.device)
        x = fc_rows(W, emb, S[r0 + 1:r1 + 1], HN[r0:r1])
        q, gate, k, v = qkv_rows(W, x, pos)
        K1[r0:r1], V1[r0:r1] = k, v
        if r1 > n - 1:
            a = max(r0, n - 1) - r0
            keep.setdefault("x", []).append(x[a:])
            keep.setdefault("q", []).append(q[a:])
            keep.setdefault("g", []).append(gate[a:])
    K1[:n], V1[:n] = Kctx[:n].float(), Vctx[:n].float()
    x, q, gate = (torch.cat(keep[k]) for k in ("x", "q", "g"))
    P = torch.arange(n, L - 1, device=S.device)            # anchors
    A = P.shape[0]
    x, q, gate = x[:A], q[:A], gate[:A]                    # rows at positions p - 1
    h = finish(W, x, q, gate, K1, V1, P)                    # sees positions <= p - 1
    out = [h]
    Kc = Vc = None
    for j in range(2, D + 1):
        m = P + j <= L - 1
        if not m.any():
            break
        P, h = P[m], h[m]
        if Kc is not None:
            Kc, Vc = Kc[m], Vc[m]
        x = fc_rows(W, emb, S[P + j - 1], h)
        q, gate, k, v = qkv_rows(W, x, P + j - 2)
        Kc = k[:, None] if Kc is None else torch.cat([Kc, k[:, None]], 1)
        Vc = v[:, None] if Vc is None else torch.cat([Vc, v[:, None]], 1)
        h = finish(W, x, q, gate, K1, V1, P, Kc, Vc)
        out.append(h)
    return out


def bits(m):
    return base64.b64encode(np.packbits(m.astype(np.uint8)).tobytes()).decode()


@torch.inference_mode()
def main():
    os.set_blocking(1, True)
    path = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    MAXT = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    D = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    S0 = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    LAYERS = os.environ.get("DS_LAYERS", "w4,w4nf,w8,bf").split(",")
    eng = Engine(MODEL)
    F, st, pre, w = eng.fast, eng.st, eng.pre, eng.w
    preb = PF.Prefill(w, fp8=False)
    t0 = time.time()
    Ws = {k: MtpW(w.mtp, k) for k in LAYERS}
    H4 = fq4(w.lm_head)                                    # 4-bit draft head rows (fp32), full vocabulary
    V = H4.shape[0]
    in_dv = torch.zeros(V, dtype=torch.bool, device="cuda")
    in_dv[torch.from_numpy(F.dv_np).cuda()] = True
    heads = {"dv4": H4, "full4": H4, "fullbf": None, "dvbf": None}
    rows = [json.loads(l) for l in open(path)][S0:N]
    GQ = int(os.environ.get("DS_GPTQ", "0"))       # number of calibration prompts (tests/prompts.jsonl) for GPTQ

    def prefill_hidden(p_, ids):
        st.ssm.zero_()
        st.conv.zero_()
        eng.cache.in_work = None
        L = ids.shape[0]
        HN = torch.empty(L, HIDDEN, dtype=torch.bfloat16, device="cuda")
        for s in range(0, L, PF.CHUNK):
            b = min(L, s + PF.CHUNK)
            HN[s:b] = p_.chunk(ids[s:b], s, st)
            m = min(b, L - 1) - s
            p_.mtp(ids[s + 1:s + 1 + m], HN[s:s + m], s, st)
        return HN, st.k[16, :L].clone(), st.v[16, :L].clone()

    ENGV = os.environ.get("DS_ENG", "1") == "1"
    pos_ = torch.zeros(1, dtype=torch.int32, device="cuda")
    nkv_ = torch.zeros(1, dtype=torch.int32, device="cuda")
    dft = torch.zeros(1, dtype=torch.long, device="cuda")

    def engine_chain(S, n, HN, Ks, Vs):
        """the engine's own MTP kernels (F.mtp, 4-bit, draft vocabulary + extra), anchor by anchor (as tree_sim);
        MTP KV of positions >= p - 1 restored to the prefill path's after each anchor"""
        L = S.shape[0]
        A = L - 1 - n
        m = torch.zeros(A, D, dtype=torch.bool, device="cuda")
        pc = torch.zeros(A, D, device="cuda")
        st.k[16, :L] = Ks
        st.v[16, :L] = Vs
        for a in range(A):
            p = n + a
            for j in range(1, D + 1):
                if p + j >= L:
                    break
                pos_.fill_(p + j - 2)
                nkv_.fill_(p + j - 1)
                hid = HN[p - 1:p] if j == 1 else F.mhn[:1]
                F.mtp(st, S[p + j - 1:p + j], hid, pos_, nkv_, 1, F.zero, dft)
                m[a, j - 1] = dft[0] == S[p + j]
                pc[a, j - 1] = F.mlogit[0].float().softmax(-1).max()
            st.k[16, p - 1:L] = Ks[p - 1:L]
            st.v[16, p - 1:L] = Vs[p - 1:L]
        return m.cpu().numpy(), (pc * 255).round().byte().cpu().numpy()

    attn2 = PF.ATTN2
    if GQ:
        Wb = Ws["bf"] if "bf" in Ws else MtpW(w.mtp, "bf")
        Hs = {}
        for r in [json.loads(l) for l in open("tests/prompts.jsonl")][:GQ]:
            ids = torch.tensor(r["prompt_token_ids"], device="cuda")
            HN = prefill_hidden(pre, ids)[0]
            calib(Wb, w.embed, ids, HN, Hs)
        Wg = MtpW(w.mtp, "gq")                     # bf16 values with the norms folded, then GPTQ per matrix
        norms = {"qkv": "in_norm", "gate_up": "post_norm"}
        for k in ("fc", "qkv", "o", "gate_up", "down"):
            Wf = w.mtp[k].float() * ((1 + w.mtp[norms[k]].float())[None] if k in norms else 1)
            setattr(Wg, k, gptq(Wf, Hs[k]))
        Ws["gq"] = Wg
        LAYERS.append("gq")
        heads["dvg4"] = gptq(w.lm_head.float(), Hs["head"])
        print(f"gptq done in {time.time() - t0:.1f} s", flush=True)
    print(f"study weights in {time.time() - t0:.1f} s", flush=True)
    for r in rows:
        prompt = r["prompt_token_ids"]
        n = len(prompt)
        out, _ = eng.generate(prompt, MAXT)
        acc = list(eng.accepted)
        out = out[:MAXT]
        SE = torch.tensor(prompt + out, device="cuda")
        # bf16 greedy path (stepwise) and bf16 hidden
        PF.ATTN2 = False
        st.ssm.zero_()
        st.conv.zero_()
        eng.cache.in_work = None
        ids = torch.tensor(prompt, device="cuda")
        HNb = torch.empty(n + MAXT, HIDDEN, dtype=torch.bfloat16, device="cuda")
        for s in range(0, n, PF.CHUNK):
            b = min(n, s + PF.CHUNK)
            HNb[s:b] = preb.chunk(ids[s:b], s, st)
        outb = []
        for i in range(MAXT):
            t = int((HNb[n + i - 1:n + i] @ w.lm_head.t()).float().argmax())
            outb.append(t)
            if t in (248046, 248044) or i == MAXT - 1:
                break
            HNb[n + i] = preb.chunk(torch.tensor([t], device="cuda"), n + i, st)[0]
        SB = torch.tensor(prompt + outb, device="cuda")
        del HNb
        # bf16 argmax along the engine path (the judge's top-1 share) and the fp8 hidden of the engine path
        HNbe = prefill_hidden(preb, SE)[0]
        yb = (HNbe[n - 1:-1] @ w.lm_head.t()).float().argmax(-1)
        top1 = float((yb == SE[n:]).float().mean())
        HNb, Kb, Vb = prefill_hidden(preb, SB)
        PF.ATTN2 = attn2
        HNe, Ke, Ve = prefill_hidden(pre, SE)
        del HNbe
        me = engine_chain(SE, n, HNe, Ke, Ve) if ENGV else None
        # prompt extra slots, as Fast.set_extra
        pt = np.asarray(prompt)
        ids_, cnt = np.unique(pt[~F.in_dv[pt]], return_counts=True)
        ids_ = ids_[np.argsort(-cnt, kind="stable")[:256]]
        sub = in_dv.clone()
        sub[torch.from_numpy(ids_).cuda()] = True
        doc = {"id": r.get("id"), "n": n, "acc": acc, "out": out, "outb": outb, "top1": top1, "D": D, "m": {}}
        if me is not None:
            doc["m"]["E-eng-dv4"] = bits(me[0])
            doc["pc"] = base64.b64encode(me[1].tobytes()).decode()
        for pname, S, HN, Kc_, Vc_ in (("E", SE, HNe, Ke, Ve), ("B", SB, HNb, Kb, Vb)):
            L = S.shape[0]
            if L - 1 - n < 1:
                continue
            for lk in LAYERS:
                if pname == "B" and lk not in ("w4", "bf", "w8"):
                    continue
                hs = chains(Ws[lk], w.embed, S, n, HN, D, Kc_, Vc_)
                for hk, HW in heads.items():
                    A = L - 1 - n
                    m = np.zeros((A, D), dtype=bool)
                    for j, h in enumerate(hs, 1):
                        if HW is not None:
                            lg = h @ HW.t()
                        else:
                            lg = (h.bfloat16() @ w.lm_head.t()).float()
                        if hk.startswith("dv"):
                            lg.masked_fill_(~sub[None], float("-inf"))
                        d = lg.argmax(-1)
                        tr = S[n + j:n + j + d.shape[0]]
                        m[:d.shape[0], j - 1] = (d == tr).cpu().numpy()
                    doc["m"][f"{pname}-{lk}-{hk}"] = bits(m)
        doc["A"] = {"E": SE.shape[0] - 1 - n, "B": SB.shape[0] - 1 - n}
        line = json.dumps(doc)
        for i in range(0, len(line), 4000):
            print("DSP " + line[i:i + 4000], flush=True)
        print("DSE", flush=True)
        print(f"{r.get('id')} n={n} gen E {len(out)} B {len(outb)} top1 {top1:.3f} engine {sum(acc) / len(acc):.2f} "
              f"t/r", flush=True)


if __name__ == "__main__":
    main()
