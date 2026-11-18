"""Prefill: the target model (and the MTP layer) over a chunk of up to CHUNK positions of one sequence, written into
the working state (attention KV, GDN states in [layer, head, v, k] layout, conv histories)."""
import torch
import torch.nn.functional as F

from . import hip as H
from . import ref_ops as R
from .model import CONV_DIM, HEAD_DIM, HEADS, HIDDEN, INTER, IS_ATTN, KV_HEADS, V_DIM, VOCAB

CHUNK = 4096
PROJ = CONV_DIM + V_DIM + 96


def gdn_chunk_ref(qk, v, gb, ssmT):
    """qk (T, 32, 128) fp32 normalized (q scaled), v (T, 48, 128), gb (T, 2, 48); ssmT (48, v, k) in place."""
    q = qk[:, :16].repeat_interleave(3, 1)
    k = qk[:, 16:].repeat_interleave(3, 1)
    S = ssmT.transpose(-1, -2).contiguous()
    o = R.gdn_chunked(q, k, v, gb[:, 0], gb[:, 1], S)
    ssmT.copy_(S.transpose(-1, -2))
    return o


def attn_prefill_ref(q, kc, vc, p0, T):
    n = p0 + T
    K = kc[:n].transpose(0, 1).repeat_interleave(HEADS // KV_HEADS, 0)
    V = vc[:n].transpose(0, 1).repeat_interleave(HEADS // KV_HEADS, 0)
    qi = torch.arange(p0, n, device=q.device)[:, None]
    kj = torch.arange(n, device=q.device)[None]
    o = F.scaled_dot_product_attention(q.transpose(0, 1), K, V, attn_mask=kj <= qi)
    return o.transpose(0, 1).reshape(T, HEADS * HEAD_DIM)


class Prefill:
    def __init__(self, w, dev="cuda", gdn_chunk=gdn_chunk_ref, attn_prefill=attn_prefill_ref):
        self.w = w
        self.gdn_chunk, self.attn_prefill = gdn_chunk, attn_prefill
        C = CHUNK
        bf = dict(dtype=torch.bfloat16, device=dev)
        self.x = torch.empty(C, HIDDEN, **bf)
        self.h = torch.empty(C, HIDDEN, **bf)
        self.d = torch.empty(C, HIDDEN, **bf)
        self.proj = torch.empty(C, PROJ, **bf)
        self.qk = torch.empty(C, 32, 128, dtype=torch.float32, device=dev)
        self.vv = torch.empty(C, 48, 128, dtype=torch.float32, device=dev)
        self.gb = torch.empty(C, 2, 48, dtype=torch.float32, device=dev)
        self.go = torch.empty(C, V_DIM, **bf)
        self.qkv = torch.empty(C, 14336, **bf)
        self.q = torch.empty(C, HEADS, HEAD_DIM, **bf)
        self.ao = torch.empty(C, HEADS * HEAD_DIM, **bf)
        self.gu = torch.empty(C, 2 * INTER, **bf)
        self.act = torch.empty(C, INTER, **bf)
        self.hn = torch.empty(C, HIDDEN, **bf)
        self.mx = torch.empty(C, 2 * HIDDEN, **bf)
        self.pos = torch.zeros(1, dtype=torch.int32, device=dev)
        self.nrow = torch.zeros(1, dtype=torch.int32, device=dev)

    def _attn(self, L, li, st, p0, T):
        torch.matmul(self.h[:T], L["qkv"].t(), out=self.qkv[:T])
        H.attn_prep(self.qkv[:T], L["q_norm"], L["k_norm"], self.pos, self.q[:T], st.k[li], st.v[li])
        o = self.attn_prefill(self.q[:T], st.k[li], st.v[li], p0, T)
        self.ao[:T].copy_(o)
        H.gate_mul(self.ao[:T], self.qkv[:T])
        torch.matmul(self.ao[:T], L["o"].t(), out=self.d[:T])

    def _mlp(self, L, T):
        H.add_rmsnorm(self.x[:T], self.d[:T], L["post_norm"], self.h[:T])
        torch.matmul(self.h[:T], L["gate_up"].t(), out=self.gu[:T])
        H.silu_mul(self.gu[:T], self.act[:T])
        torch.matmul(self.act[:T], L["down"].t(), out=self.d[:T])

    @torch.inference_mode()
    def chunk(self, ids, p0, st):
        """ids (T,) long on the device at positions p0..; returns the final-normed hidden (T, HIDDEN) view."""
        w, T = self.w, ids.shape[0]
        self.pos.fill_(p0)
        self.nrow.fill_(T)
        torch.index_select(w.embed, 0, ids, out=self.x[:T])
        ai = gi = 0
        for i, L in enumerate(w.layers):
            H.add_rmsnorm(self.x[:T], None if i == 0 else self.d[:T], L["in_norm"], self.h[:T])
            if IS_ATTN[i]:
                self._attn(L, ai, st, p0, T)
                ai += 1
            else:
                proj = self.proj[:T]
                torch.matmul(self.h[:T], L["in_proj"].t(), out=proj)
                H.gdn_prep(proj, st.conv[gi], L["conv"], L["A_neg"], L["dt_bias"], self.qk[:T], self.vv[:T],
                           self.gb[:T])
                o = self.gdn_chunk(self.qk[:T], self.vv[:T], self.gb[:T], st.ssm[gi])
                H.gated_norm(o, proj, L["gnorm"], self.go[:T])
                H.conv_update(self.nrow, proj, st.conv[gi])
                torch.matmul(self.go[:T], L["out"].t(), out=self.d[:T])
                gi += 1
            self._mlp(L, T)
        H.add_rmsnorm(self.x[:T], self.d[:T], w.final_norm, self.hn[:T])
        return self.hn[:T]

    def logits_last(self, hn):
        return torch.matmul(hn[-1:], self.w.lm_head.t()).float()

    @torch.inference_mode()
    def mtp(self, next_ids, hn, p0, st):
        """The MTP layer over the chunk: row i = (hidden at p0+i, token next_ids[i] at p0+i+1). Writes MTP KV.
        Returns the last row's logits and normed hidden."""
        w, M, T = self.w, self.w.mtp, next_ids.shape[0]
        self.pos.fill_(p0)
        torch.index_select(w.embed, 0, next_ids, out=self.h[:T])
        H.add_rmsnorm(self.h[:T], None, M["pre_emb"], self.mx[:T, :HIDDEN])
        H.add_rmsnorm(hn, None, M["pre_hid"], self.mx[:T, HIDDEN:])
        torch.matmul(self.mx[:T], M["fc"].t(), out=self.x[:T])
        H.add_rmsnorm(self.x[:T], None, M["in_norm"], self.h[:T])
        self._attn(M, 16, st, p0, T)
        self._mlp(M, T)
        out = torch.empty(1, HIDDEN, dtype=torch.bfloat16, device=hn.device)
        H.add_rmsnorm(self.x[T - 1:T], self.d[T - 1:T], M["norm"], out)
        return torch.matmul(out, w.lm_head.t()).float(), out
