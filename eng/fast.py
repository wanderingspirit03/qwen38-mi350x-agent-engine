"""Decode path: speculative verification of R = K + 1 positions and MTP drafting, with fixed shapes and all
positions on the device, so a whole round can be captured in one HIP graph.

Round (all on the device):
  verify   the target model on [x, d_1..d_K] at positions pos..pos+K -> y = argmax per row
  accept   a = number of leading drafts equal to y; n = a + 1 new tokens: d_1..d_a, y_a
  commit   GDN states and conv histories advance by n rows
  catch-up the MTP layer on rows (target hidden at pos+i, token at pos+i+1), i = 0..K; row a gives d'_1
  chain    K - 1 single-row MTP steps give d'_2..d'_K
  pos += n, x = y_a
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import hip as H
from .kernels import w4 as w4q
from .kernels.attention import attn_decode
from .model import (CONV_DIM, HEAD_DIM, HEADS, HIDDEN, INTER, IS_ATTN, KV_HEADS, N_LAYERS, V_DIM, VOCAB)

PROJ = CONV_DIM + V_DIM + 96


class SeqState:
    """The decode-side state of one sequence: attention KV (16 target layers + the MTP layer), GDN recurrent
    states in [layer, head, v, k] layout, conv histories."""

    def __init__(self, max_len, dev="cuda"):
        self.k = torch.zeros(17, max_len, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        self.v = torch.zeros_like(self.k)
        self.ssm = torch.zeros(48, 48, 128, 128, dtype=torch.float32, device=dev)
        self.conv = torch.zeros(48, 3, CONV_DIM, dtype=torch.bfloat16, device=dev)
        self.max_len = max_len


W4_HEAD = os.environ.get("ENG_W4_HEAD", "1") == "1"


class Linear:
    """x (M <= 16, K) @ W^T on 4-bit weights (eng.kernels.w4). norm: the RMSNorm weight in front of W, folded in
    (x is then the raw residual stream); swiglu: W is [gate; up] and the output is silu(gate) * up;
    add=True: out += x @ W^T (out is the residual stream)."""

    def __init__(self, w, norm=None, swiglu=False):
        self.p = w4q.quantize(w, norm_weight=norm, swiglu=swiglu)

    def __call__(self, x, out, add=False):
        w4q.gemm(x, self.p, out=out, residual=out if add else None)


class Bf16Linear:
    def __init__(self, w):
        self.w = w

    def __call__(self, x, out):
        torch.matmul(x, self.w.t(), out=out)


def attn_decode_ref(q, kc, vc, pos_dev, R):
    """q (R, 24, 256) at positions pos..pos+R-1; keys 0..pos+R-1. Eager (host reads pos)."""
    n = int(pos_dev.item()) + R
    K = kc[:n].transpose(0, 1).repeat_interleave(HEADS // KV_HEADS, 0)
    V = vc[:n].transpose(0, 1).repeat_interleave(HEADS // KV_HEADS, 0)
    qi = torch.arange(n - R, n, device=q.device)[:, None]
    kj = torch.arange(n, device=q.device)[None]
    o = F.scaled_dot_product_attention(q.transpose(0, 1), K, V, attn_mask=kj <= qi)
    return o.transpose(0, 1).reshape(R, HEADS * HEAD_DIM)


class Fast:
    def __init__(self, w, K, dev="cuda"):
        self.w, self.K, self.R = w, K, K + 1
        R = self.R
        bf = dict(dtype=torch.bfloat16, device=dev)
        self.lin = []
        for i, L in enumerate(w.layers):
            self.lin.append(self._layer_lin(L, IS_ATTN[i]))
        M = w.mtp
        self.mlin = self._layer_lin(M, True)
        self.mlin["fc"] = Linear(M["fc"])
        HL = Linear if W4_HEAD else Bf16Linear
        self.head = HL(w.lm_head)
        # drafts are chosen from the 32768 most frequent output tokens (eng/draft_vocab.npy): an 8x smaller head
        dv = np.load(os.path.join(os.path.dirname(__file__), "draft_vocab.npy"))
        self.dvocab = torch.from_numpy(dv.astype(np.int64)).to(dev)
        self.dhead = HL(w.lm_head[self.dvocab].contiguous())
        # GDN per-layer weights stacked for the all-layer commit
        G = [L for i, L in enumerate(w.layers) if not IS_ATTN[i]]
        self.cw = torch.stack([L["conv"] for L in G])
        self.aneg = torch.stack([L["A_neg"] for L in G]).contiguous()
        self.dtb = torch.stack([L["dt_bias"] for L in G]).contiguous()
        # buffers
        self.x = torch.empty(R, HIDDEN, **bf)
        self.proj = torch.empty(48, R, PROJ, **bf)
        self.qk = torch.empty(48, R, 32, 128, dtype=torch.float32, device=dev)
        self.vv = torch.empty(48, R, 48, 128, dtype=torch.float32, device=dev)
        self.gb = torch.empty(48, R, 2, 48, dtype=torch.float32, device=dev)
        self.go = torch.empty(R, V_DIM, **bf)
        self.ob = torch.empty(R, 48, 128, dtype=torch.float32, device=dev)
        self.qkv = torch.empty(R, 14336, **bf)
        self.q = torch.empty(R, HEADS, HEAD_DIM, **bf)
        self.ao = torch.empty(R, HEADS * HEAD_DIM, **bf)
        self.act = torch.empty(R, INTER, **bf)
        self.logits = torch.empty(R, VOCAB, **bf)
        self.hn = torch.empty(R, HIDDEN, **bf)          # final-normed hidden (MTP input)
        self.mx = torch.empty(R, 2 * HIDDEN, **bf)
        self.mh = torch.empty(R, HIDDEN, **bf)
        self.mlogit = torch.empty(1, len(dv), **bf)
        self.mhn = torch.empty(R, HIDDEN, **bf)
        # round state on the device
        i32 = dict(dtype=torch.int32, device=dev)
        self.pos = torch.zeros(1, **i32)                  # position of x
        self.vt = torch.zeros(R, dtype=torch.long, device=dev)   # [x, d_1..d_K]
        self.y = torch.zeros(R, dtype=torch.long, device=dev)
        self.n_acc = torch.ones(1, **i32)
        self.new = torch.zeros(R, dtype=torch.long, device=dev)  # the round's new tokens (first n_acc valid)
        self.pick = torch.zeros(1, dtype=torch.long, device=dev)
        # positions derived from pos (see accept_kernel): [catch-up pos, its keys, verify keys, (chain pos, keys)...]
        self.P = torch.zeros(3 + 2 * K, **i32)
        self.res = torch.zeros(R + 1, dtype=torch.long, device=dev)   # [n_acc, new tokens...]
        self.zero = torch.zeros(1, dtype=torch.long, device=dev)

    # ---- the target model on R rows ----
    @staticmethod
    def _layer_lin(L, attn):
        d = {"gate_up": Linear(L["gate_up"], L["post_norm"], swiglu=True), "down": Linear(L["down"])}
        if attn:
            d["qkv"], d["o"] = Linear(L["qkv"], L["in_norm"]), Linear(L["o"])
        else:
            d["in_proj"], d["out"] = Linear(L["in_proj"], L["in_norm"]), Linear(L["out"])
        return d

    def mlp(self, lin, rows):
        """x += down(swiglu(gate_up(norm(x))))"""
        R = rows
        lin["gate_up"](self.x[:R], self.act[:R])
        lin["down"](self.act[:R], self.x[:R], add=True)

    def verify(self, st):
        """self.vt at positions self.pos.. -> self.y (argmax per row), self.hn."""
        w, R = self.w, self.R
        torch.index_select(w.embed, 0, self.vt, out=self.x)
        ai = gi = 0
        for i, L in enumerate(w.layers):
            lin = self.lin[i]
            if IS_ATTN[i]:
                self.attn(lin, L, ai, st, self.pos, self.P[2:3], R)
                ai += 1
            else:
                proj = self.proj[gi]
                lin["in_proj"](self.x, proj)
                H.gdn_prep(proj, st.conv[gi], L["conv"], L["A_neg"], L["dt_bias"], self.qk[gi], self.vv[gi],
                           self.gb[gi])
                H.gdn_verify(self.qk[gi], self.vv[gi], self.gb[gi], st.ssm[gi], self.ob, proj, L["gnorm"], self.go)
                lin["out"](self.go, self.x, add=True)
                gi += 1
            self.mlp(lin, R)
        H.add_rmsnorm(self.x, None, w.final_norm, self.hn)
        self.head(self.hn, self.logits)
        H.argmax_rows(self.logits, self.y)

    def set_pos(self, n):
        """Start of decoding at position n (eager)."""
        self.pos.fill_(n)
        self.P[2] = n + self.R
        self.P[3::2] = torch.arange(n, n + self.K, dtype=torch.int32)
        self.P[4::2] = torch.arange(n + 1, n + self.K + 1, dtype=torch.int32)

    def attn(self, lin, L, li, st, pos, nkv, R):
        lin["qkv"](self.x[:R], self.qkv[:R])
        H.attn_prep(self.qkv[:R], L["q_norm"], L["k_norm"], pos, self.q[:R], st.k[li], st.v[li])
        gate = self.qkv[:R, :HEADS * 512].view(R, HEADS, 512)[..., HEAD_DIM:]
        attn_decode(self.q[:R], st.k[li], st.v[li], nkv, gate=gate, out=self.ao[:R].view(R, HEADS, HEAD_DIM))
        lin["o"](self.ao[:R], self.x[:R], add=True)

    # ---- acceptance and commit ----
    def accept(self, st):
        H.accept(self.vt, self.y, self.n_acc, self.new, self.res, self.pick, self.pos, self.P)
        H.gdn_commit(self.n_acc, self.qk, self.vv, self.gb, st.ssm, self.proj, st.conv)

    # ---- MTP ----
    def mtp(self, st, tokens, hidden, pos, nkv, rows, pick, out):
        """The MTP layer on `rows` rows at positions pos.. (row i: target-side hidden at pos+i and the token at
        pos+i+1). Writes the draft (argmax of row `pick`) to out (1,) and leaves that row's normed hidden
        in self.mhn[:1]."""
        w, M, R = self.w, self.w.mtp, rows
        H.mtp_in(tokens, w.embed, hidden, M["pre_emb"], M["pre_hid"], self.mx[:R])
        self.mlin["fc"](self.mx[:R], self.x[:R])
        self.attn(self.mlin, M, 16, st, pos, nkv, R)
        self.mlp(self.mlin, R)
        if R == 1:
            H.add_rmsnorm(self.x[:1], None, M["norm"], self.mhn[:1])
        else:
            H.add_rmsnorm(self.x[:R], None, M["norm"], self.mh[:R])
            self.mhn[:1].copy_(self.mh[:R].index_select(0, pick))
        self.dhead(self.mhn[:1], self.mlogit)
        H.argmax_rows(self.mlogit, out, self.dvocab)

    def draft_chain(self, st):
        """vt[1] holds d_1: fill vt[2:] with K - 1 chained MTP drafts (position of x is self.pos)."""
        for j in range(1, self.K):
            P = self.P[3 + 2 * (j - 1):]
            self.mtp(st, self.vt[j:j + 1], self.mhn[:1], P[0:1], P[1:2], 1, self.zero, self.vt[j + 1:j + 2])

    def round(self, st):
        self.verify(st)
        self.accept(st)
        # catch-up: rows i = 0..K: hidden at pos+i, token new[i] at pos+i+1; row a = n_acc - 1 gives d_1
        self.mtp(st, self.new, self.hn, self.P[0:1], self.P[1:2], self.R, self.pick, self.vt[1:2])
        self.draft_chain(st)

    def capture(self, st):
        """Capture the round and the draft chain as HIP graphs (state buffers are fixed; contents garbage)."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        self.pos.fill_(1000)
        with torch.cuda.stream(s):
            self.round(st)
            self.draft_chain(st)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.g_round = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_round):
            self.round(st)
        self.g_chain = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_chain):
            self.draft_chain(st)
        torch.cuda.synchronize()
