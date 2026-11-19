"""GPTQ for the 4-bit MTP layer and draft head (eng.kernels.w4 format: asymmetric 4-bit per group of 128, fp16
scale, zero point; the group's scale and zero come from rtn_search on the error-updated group).

Calibration: eng/calib.npz = 9 agentic prompts (6 first-turn, 3 later-turn up to 12000 tokens; repos outside all
test sets) + the engine's own greedy answers; the shared 5957-token system prompt is counted once (31539 counted
positions). The target (prefill path) over it, then the MTP layer in fp32 PyTorch on its bf16 weights: X^T X of
every MTP linear's input and of the MTP output (draft head input); TargetHessians for the target's GEMMs. On thinking prose (tests/fprompts.jsonl) this raises tokens per round by ~3.5% over round-to-nearest
(engine-notes/draft.md); the run-time cost is zero."""
import os

import numpy as np
import torch
import torch.nn.functional as F

from .kernels import w4 as w4q
from .model import HEAD_DIM, HEADS, INTER, KV_HEADS, apply_rope, rope_cos_sin

EPS = 1e-6
ACT = os.environ.get("ENG_GPTQ_ACT", "0") == "1"


def gptq_codes(W, H, G=128, damp=0.01, act=None):
    """W (N, K) fp32 (norm weight already folded), H = X^T X (K, K) fp32 -> codes u (N, K/G, G) int32 in 0..15,
    scales s and zero points z (N, K/G) fp32, as w4.rtn_search returns them.
    act (default ENG_GPTQ_ACT): columns in order of decreasing H diagonal, with the groups' scales and zero points
    fixed beforehand by rtn_search on W ("static groups"); else column order, each group's scale and zero from
    rtn_search on the error-updated group."""
    act = ACT if act is None else act
    assert G == 128
    W = W.float().clone()
    N, K = W.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    if act:
        _, s, z = w4q.rtn_search(W.reshape(N, K // G, G))
        perm = torch.argsort(torch.diag(H), descending=True)
        W, H = W[:, perm], H[perm][:, perm]
        gcol = (perm // G).tolist()         # group of each (permuted) column
    else:
        s = torch.empty(N, K // G, device=W.device)
        z = torch.empty_like(s)
    H += torch.eye(K, device=H.device) * damp * torch.diag(H).mean()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    U = torch.empty(N, K, dtype=torch.int32, device=W.device)
    B = 128
    for i1 in range(0, K, B):
        i2 = i1 + B
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        if not act:
            _, sg, zg = w4q.rtn_search(W1[:, None, :])
            s[:, i1 // G], z[:, i1 // G] = sg[:, 0], zg[:, 0]
        for i in range(B):
            g = gcol[i1 + i] if act else i1 // G
            sg, zg = s[:, g], z[:, g]
            w = W1[:, i]
            q = ((w / sg).round() + zg).clamp(0, 15)
            U[:, i1 + i] = q.int()
            e = (w - (q - zg) * sg) / Hi[i, i]
            W1[:, i:] -= e[:, None] * Hi[i, i:][None]
            E1[:, i] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    if act:
        U = U[:, torch.argsort(perm)]
    return U.reshape(N, K // G, G), s, z


def _rms(x, w=None):
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return x if w is None else x * (1.0 + w.float())


@torch.inference_mode()
def mtp_hessians(w, pre, st):
    """X^T X per MTP linear ('fc', 'qkv', 'o', 'gate_up', 'down'; inputs of qkv / gate_up are x / rms(x), the
    norm weight being folded into W) and of the draft head ('head'), over the calibration tokens."""
    M = w.mtp
    Wf = {k: M[k].float() for k in ("fc", "qkv", "o", "gate_up", "down")}
    Hs = {}

    for seq, skip in _calib_ids():
        L = seq.shape[0]
        st.ssm.zero_()
        st.conv.zero_()
        hn = torch.empty(L, w.embed.shape[1], dtype=torch.bfloat16, device="cuda")
        for s0 in range(0, L, 8192):
            hn[s0:s0 + 8192] = pre.chunk(seq[s0:s0 + 8192], s0, st)
        K1 = torch.empty(L - 1, KV_HEADS, HEAD_DIM, device="cuda")
        V1 = torch.empty_like(K1)
        rows = []
        for r0 in range(0, L - 1, 2048):
            r1 = min(L - 1, r0 + 2048)
            pos = torch.arange(r0, r1, device="cuda")
            xin = torch.cat([_rms(w.embed[seq[r0 + 1:r1 + 1]], M["pre_emb"]), _rms(hn[r0:r1], M["pre_hid"])], -1)
            x = xin @ Wf["fc"].t()
            qkv = _rms(x, M["in_norm"]) @ Wf["qkv"].t()
            R = r1 - r0
            qg = qkv[:, :HEADS * 512].view(R, HEADS, 512)
            q, gate = qg[..., :HEAD_DIM], qg[..., HEAD_DIM:].reshape(R, -1)
            k = qkv[:, HEADS * 512:HEADS * 512 + KV_HEADS * HEAD_DIM].view(R, KV_HEADS, HEAD_DIM)
            v = qkv[:, HEADS * 512 + KV_HEADS * HEAD_DIM:].view(R, KV_HEADS, HEAD_DIM)
            cos, sin = rope_cos_sin(pos, "cuda")
            q = apply_rope(_rms(q, M["q_norm"]), cos, sin)
            K1[r0:r1] = apply_rope(_rms(k, M["k_norm"]), cos, sin)
            V1[r0:r1] = v
            rows.append((r0, r1, xin, x, q, gate))
        G = HEADS // KV_HEADS
        for r0, r1, xin, x, q, gate in rows:
            if r1 <= skip:
                continue
            a0 = max(0, skip - r0)          # rows (positions) before skip are not counted

            def acc(name, X):
                X = X[a0:]
                Hs[name] = Hs.get(name, 0) + X.t() @ X

            acc("fc", xin)
            acc("qkv", _rms(x))
            qq = q.reshape(r1 - r0, KV_HEADS, G, HEAD_DIM).transpose(0, 1)            # (4, R, 6, 256)
            sc = torch.einsum("krgd,pkd->krgp", qq * HEAD_DIM ** -0.5, K1[:r1])
            sc.masked_fill_(torch.arange(r1, device="cuda")[None, :] > torch.arange(r0, r1, device="cuda")[:, None, None],
                            float("-inf"))
            a = torch.einsum("krgp,pkd->rkgd", sc.softmax(-1), V1[:r1]).reshape(r1 - r0, -1)
            a = a * torch.sigmoid(gate)
            del sc
            acc("o", a)
            x = x + a @ Wf["o"].t()
            acc("gate_up", _rms(x))
            gu = _rms(x, M["post_norm"]) @ Wf["gate_up"].t()
            act = F.silu(gu[:, :INTER]) * gu[:, INTER:]
            acc("down", act)
            x = x + act @ Wf["down"].t()
            acc("head", _rms(x, M["norm"]))
    st.ssm.zero_()
    st.conv.zero_()
    return Hs


class _Hook:
    """Stands in for a bf16 prefill Lin and adds X^T X of its input to H[name] (while _Hook.on)."""
    on = True

    def __init__(self, lin, H, name):
        self.lin, self.H, self.name, self.fp8 = lin, H, name, False

    def __call__(self, x, out):
        if _Hook.on:
            X = x.float()
            self.H[self.name] = self.H.get(self.name, 0) + X.t() @ X
        self.lin(x, out)

    def norm_in(self, x, d, w, h, out):
        from . import hip as Hk
        Hk.add_rmsnorm(x, d, w, h)
        self(h, out)


class TargetHessians:
    """X^T X of the input of every target-model GEMM as the W4A8 GEMMs see it (normed input including the norm
    weight; gated attention / GDN output; SwiGLU output), from the bf16 prefill path over the calibration tokens.
    GROUP layers per calibration pass, computed when layer() first asks for one of them (bounded memory)."""
    GROUP = 16

    def __init__(self, w, st):
        from .prefill import Prefill
        self.w, self.st = w, st
        self.pre = Prefill(w, fp8=False)
        self.ids = _calib_ids()
        self.g, self.cur = -1, {}

    def layer(self, i):
        g = i // self.GROUP
        if g != self.g:
            self.g, self.cur = g, {}
            torch.cuda.empty_cache()
            self._pass(range(g * self.GROUP, min(len(self.w.layers), (g + 1) * self.GROUP)))
        return self.cur.pop(i)

    @torch.inference_mode()
    def _pass(self, layers):
        from . import prefill as PF
        saved = {li: dict(self.pre.lin[li]) for li in layers}
        for li in layers:
            self.cur[li] = {}
            for k, lin in saved[li].items():
                self.pre.lin[li][k] = _Hook(lin, self.cur[li], k)
        attn2, PF.ATTN2 = PF.ATTN2, False
        st = self.st
        for seq, skip in self.ids:
            st.ssm.zero_()
            st.conv.zero_()
            for a, b, on in ((0, skip, False), (skip, seq.shape[0], True)):
                _Hook.on = on
                for s0 in range(a, b, PF.CHUNK):
                    self.pre.chunk(seq[s0:min(b, s0 + PF.CHUNK)], s0, st)
        _Hook.on = True
        PF.ATTN2 = attn2
        st.ssm.zero_()
        st.conv.zero_()
        for li in layers:
            self.pre.lin[li].update(saved[li])


def _calib_ids():
    """[(token ids on the device, skip)]: positions < skip are computed (context) but not counted"""
    c = np.load(os.path.join(os.path.dirname(__file__), "calib.npz"))
    ids = torch.from_numpy(c["ids"].astype(np.int64)).cuda()
    return [(s, int(k)) for s, k in zip(torch.split(ids, c["lens"].tolist()), c["skips"])]
