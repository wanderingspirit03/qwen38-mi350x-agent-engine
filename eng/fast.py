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
import torch
import torch.nn.functional as F

from . import hip as H
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


class Linear:
    """Interim: bf16 matmul. To be replaced by the 4-bit kernel."""

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
            d = {"gate_up": Linear(L["gate_up"]), "down": Linear(L["down"])}
            if IS_ATTN[i]:
                d["qkv"], d["o"] = Linear(L["qkv"]), Linear(L["o"])
            else:
                d["in_proj"], d["out"] = Linear(L["in_proj"]), Linear(L["out"])
            self.lin.append(d)
        M = w.mtp
        self.mlin = {k: Linear(M[k]) for k in ("fc", "qkv", "o", "gate_up", "down")}
        self.head = Linear(w.lm_head)
        # GDN per-layer weights stacked for the all-layer commit
        G = [L for i, L in enumerate(w.layers) if not IS_ATTN[i]]
        self.cw = torch.stack([L["conv"] for L in G])
        self.aneg = torch.stack([L["A_neg"] for L in G]).contiguous()
        self.dtb = torch.stack([L["dt_bias"] for L in G]).contiguous()
        # buffers
        self.x = torch.empty(R, HIDDEN, **bf)
        self.h = torch.empty(R, HIDDEN, **bf)
        self.d = torch.empty(R, HIDDEN, **bf)
        self.proj = torch.empty(48, R, PROJ, **bf)
        self.qk = torch.empty(48, R, 32, 128, dtype=torch.float32, device=dev)
        self.vv = torch.empty(48, R, 48, 128, dtype=torch.float32, device=dev)
        self.gb = torch.empty(48, R, 2, 48, dtype=torch.float32, device=dev)
        self.go = torch.empty(R, V_DIM, **bf)
        self.ob = torch.empty(R, 48, 128, dtype=torch.float32, device=dev)
        self.qkv = torch.empty(R, 14336, **bf)
        self.q = torch.empty(R, HEADS, HEAD_DIM, **bf)
        self.ao = torch.empty(R, HEADS * HEAD_DIM, **bf)
        self.gu = torch.empty(R, 2 * INTER, **bf)
        self.act = torch.empty(R, INTER, **bf)
        self.logits = torch.empty(R, VOCAB, **bf)
        self.hn = torch.empty(R, HIDDEN, **bf)          # final-normed hidden (MTP input)
        self.mx = torch.empty(R, 2 * HIDDEN, **bf)
        self.mh = torch.empty(R, HIDDEN, **bf)
        self.mlogit = torch.empty(1, VOCAB, **bf)
        self.mhn = torch.empty(R, HIDDEN, **bf)
        # round state on the device
        i32 = dict(dtype=torch.int32, device=dev)
        self.pos = torch.zeros(1, **i32)                  # position of x
        self.vt = torch.zeros(R, dtype=torch.long, device=dev)   # [x, d_1..d_K]
        self.y = torch.zeros(R, dtype=torch.long, device=dev)
        self.n_acc = torch.ones(1, **i32)
        self.new = torch.zeros(R, dtype=torch.long, device=dev)  # the round's new tokens (first n_acc valid)
        self.mpos = torch.zeros(1, **i32)
        self.nkv = torch.zeros(1, **i32)
        self.res = torch.zeros(R + 1, dtype=torch.long, device=dev)   # [n_acc, new tokens...]
        self.ar = torch.arange(R, device=dev)
        self.zero = torch.zeros(1, dtype=torch.long, device=dev)

    # ---- the target model on R rows ----
    def mlp(self, lin, L, rows):
        R = rows
        H.add_rmsnorm(self.x[:R], self.d[:R], L["post_norm"], self.h[:R])
        lin["gate_up"](self.h[:R], self.gu[:R])
        H.silu_mul(self.gu[:R], self.act[:R])
        lin["down"](self.act[:R], self.d[:R])

    def verify(self, st):
        """self.vt at positions self.pos.. -> self.y (argmax per row), self.hn."""
        w, R = self.w, self.R
        torch.index_select(w.embed, 0, self.vt, out=self.x)
        self.d.zero_()
        ai = gi = 0
        for i, L in enumerate(w.layers):
            lin = self.lin[i]
            H.add_rmsnorm(self.x, self.d, L["in_norm"], self.h)
            if IS_ATTN[i]:
                self.attn(lin, L, ai, st, self.pos, R)
                ai += 1
            else:
                proj = self.proj[gi]
                lin["in_proj"](self.h, proj)
                H.gdn_prep(proj, st.conv[gi], L["conv"], L["A_neg"], L["dt_bias"], self.qk[gi], self.vv[gi],
                           self.gb[gi])
                H.gdn_verify(self.qk[gi], self.vv[gi], self.gb[gi], st.ssm[gi], self.ob, proj, L["gnorm"], self.go)
                lin["out"](self.go, self.d)
                gi += 1
            self.mlp(lin, L, R)
        H.add_rmsnorm(self.x, self.d, w.final_norm, self.hn)
        self.head(self.hn, self.logits)
        torch.argmax(self.logits, -1, out=self.y)

    def attn(self, lin, L, li, st, pos, R):
        lin["qkv"](self.h[:R], self.qkv[:R])
        H.attn_prep(self.qkv[:R], L["q_norm"], L["k_norm"], pos, self.q[:R], st.k[li], st.v[li])
        torch.add(pos, R, out=self.nkv)
        gate = self.qkv[:R, :HEADS * 512].view(R, HEADS, 512)[..., HEAD_DIM:]
        attn_decode(self.q[:R], st.k[li], st.v[li], self.nkv, gate=gate, out=self.ao[:R].view(R, HEADS, HEAD_DIM))
        lin["o"](self.ao[:R], self.d[:R])

    # ---- acceptance and commit ----
    def accept(self, st):
        K = self.K
        match = (self.vt[1:] == self.y[:K]).int()
        a = torch.cumprod(match, 0).sum()
        self.n_acc.copy_((a + 1).view(1))
        # new tokens: d_1..d_a, then y_a
        ya = self.y.gather(0, a.view(1))
        self.new.copy_(torch.where(self.ar < a, torch.cat([self.vt[1:], self.vt[:1]]), ya))
        H.gdn_commit(self.n_acc, self.qk, self.vv, self.gb, st.ssm, self.proj, st.conv)

    # ---- MTP ----
    def mtp(self, st, tokens, hidden, pos, rows, pick):
        """The MTP layer on `rows` rows at positions pos.. (row i: target-side hidden at pos+i and the token at
        pos+i+1). Returns the argmax of row `pick` (device index, shape (1,)) and leaves that row's normed hidden
        in self.mhn[:1]."""
        w, M, R = self.w, self.w.mtp, rows
        torch.index_select(w.embed, 0, tokens, out=self.mh[:R])
        H.add_rmsnorm(self.mh[:R], None, M["pre_emb"], self.mx[:R, :HIDDEN])
        H.add_rmsnorm(hidden, None, M["pre_hid"], self.mx[:R, HIDDEN:])
        self.mlin["fc"](self.mx[:R], self.x[:R])
        H.add_rmsnorm(self.x[:R], None, M["in_norm"], self.h[:R])
        self.attn(self.mlin, M, 16, st, pos, R)
        self.mlp(self.mlin, M, R)
        H.add_rmsnorm(self.x[:R], self.d[:R], M["norm"], self.mh[:R])
        row = self.mh[:R].index_select(0, pick)
        self.mhn[:1].copy_(row)
        self.head(self.mhn[:1], self.mlogit)
        return torch.argmax(self.mlogit, -1)

    def draft_chain(self, st):
        """vt[1] holds d_1: fill vt[2:] with K - 1 chained MTP drafts (position of x is self.pos)."""
        for j in range(1, self.K):
            self.mpos.copy_(self.pos + (j - 1))
            dj = self.mtp(st, self.vt[j:j + 1], self.mhn[:1], self.mpos, 1, self.zero)
            self.vt[j + 1:j + 2].copy_(dj)

    def round(self, st):
        self.verify(st)
        self.accept(st)
        # catch-up: rows i = 0..K: hidden at pos+i, token new[i] at pos+i+1; row a = n_acc - 1 gives d_1
        d1 = self.mtp(st, self.new, self.hn, self.pos, self.R, self.n_acc.long() - 1)
        self.vt[0:1].copy_(self.new.gather(0, self.n_acc.long() - 1))
        self.vt[1:2].copy_(d1)
        self.res[0:1].copy_(self.n_acc)
        self.res[1:].copy_(self.new)
        self.pos.add_(self.n_acc)
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
