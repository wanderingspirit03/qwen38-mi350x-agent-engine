"""Prefill: the target model (and the MTP layer) over a chunk of up to CHUNK positions of one sequence, written into
the working state (attention KV, GDN states in [layer, head, v, k] layout, conv histories)."""
import torch
import torch.nn.functional as F

from . import hip as H
from . import ref_ops as R
from .kernels.attention import attn_prefill
from .kernels.gdn_prefill import gdn_prefill
from .model import CONV_DIM, HEAD_DIM, HEADS, HIDDEN, INTER, IS_ATTN, KV_HEADS, V_DIM, VOCAB

CHUNK = 8192
PROJ = CONV_DIM + V_DIM + 96


def gdn_chunk_ref(qk, v, gb, ssmT):
    """qk (T, 32, 128) fp32 normalized (q scaled), v (T, 48, 128), gb (T, 2, 48); ssmT (48, v, k) in place."""
    q = qk[:, :16].repeat_interleave(3, 1)
    k = qk[:, 16:].repeat_interleave(3, 1)
    S = ssmT.transpose(-1, -2).contiguous()
    o = R.gdn_chunked(q, k, v, gb[:, 0], gb[:, 1], S)
    ssmT.copy_(S.transpose(-1, -2))
    return o.contiguous()


def attn_prefill_ref(q, kc, vc, p0, T):
    n = p0 + T
    K = kc[:n].transpose(0, 1).repeat_interleave(HEADS // KV_HEADS, 0)
    V = vc[:n].transpose(0, 1).repeat_interleave(HEADS // KV_HEADS, 0)
    qi = torch.arange(p0, n, device=q.device)[:, None]
    kj = torch.arange(n, device=q.device)[None]
    o = F.scaled_dot_product_attention(q.transpose(0, 1), K, V, attn_mask=kj <= qi)
    return o.transpose(0, 1).reshape(T, HEADS * HEAD_DIM)


FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


_QBUF = torch.empty(CHUNK * 17408, dtype=FP8, device="cuda")      # activations of one chunk, quantized
_SBUF = torch.empty(CHUNK, 1, dtype=torch.float32, device="cuda")


class Lin:
    """A prefill linear layer: bf16 matmul, or fp8 (per-output-channel weight scales, per-row activation scales)."""

    def __init__(self, w, fp8):
        self.fp8 = fp8
        if fp8:
            s = w.float().abs().amax(1, keepdim=True).clamp(min=1e-8) / FP8_MAX
            self.wq = (w.float() / s).to(FP8)
            self.ws = s.reshape(1, -1).contiguous()
        else:
            self.w = w

    def __call__(self, x, out):
        if not self.fp8:
            torch.matmul(x, self.w.t(), out=out)
            return
        M, K = x.shape
        xq = _QBUF[: M * K].view(M, K)
        H.quant_fp8(x, xq, _SBUF[:M])
        self.mm_q(xq, out)

    def norm_in(self, x, d, w, h, out):
        """out = rmsnorm(x += d) @ W^T; h: bf16 scratch for the normed rows when not fp8."""
        T = x.shape[0]
        if self.fp8:
            xq = _QBUF[: T * HIDDEN].view(T, HIDDEN)
            H.add_rmsnorm_q(x, d, w, xq, _SBUF[:T])
            self.mm_q(xq, out)
        else:
            H.add_rmsnorm(x, d, w, h)
            self(h, out)

    def mm_q(self, xq, out):
        """xq: rows already quantized into _QBUF, scales in _SBUF."""
        torch._scaled_mm(xq, self.wq.t(), scale_a=_SBUF[:xq.shape[0]], scale_b=self.ws, out_dtype=torch.bfloat16,
                         out=out)


class Prefill:
    def __init__(self, w, dev="cuda", gdn_chunk=gdn_chunk_ref, fp8=False):
        self.w = w
        self.gdn_chunk = gdn_chunk
        self.lin = []
        for L in w.layers:
            self.lin.append({k: Lin(L[k], fp8) for k in ("qkv", "o", "in_proj", "out", "gate_up", "down") if k in L})
        self.mlin = {k: Lin(w.mtp[k], fp8) for k in ("fc", "qkv", "o", "gate_up", "down")}
        C = CHUNK
        bf = dict(dtype=torch.bfloat16, device=dev)
        self.x = torch.empty(C, HIDDEN, **bf)
        self.h = torch.empty(C, HIDDEN, **bf)
        self.d = torch.empty(C, HIDDEN, **bf)
        self.proj = torch.empty(C, PROJ, **bf)
        self.y = torch.empty(C, CONV_DIM, **bf)
        self.o = torch.empty(C, 48, 128, **bf)
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

    def _attn(self, L, lin, li, st, p0, T, d):
        lin["qkv"].norm_in(self.x[:T], d, L["in_norm"], self.h[:T], self.qkv[:T])
        H.attn_prep(self.qkv[:T], L["q_norm"], L["k_norm"], self.pos, self.q[:T], st.k[li], st.v[li])
        gate = self.qkv[:T, :HEADS * 512].view(T, HEADS, 512)[..., HEAD_DIM:]
        attn_prefill(self.q[:T], st.k[li], st.v[li], p0, gate=gate, out=self.ao[:T].view(T, HEADS, HEAD_DIM))
        lin["o"](self.ao[:T], self.d[:T])

    def _mlp(self, L, lin, T):
        lin["gate_up"].norm_in(self.x[:T], self.d[:T], L["post_norm"], self.h[:T], self.gu[:T])
        if lin["down"].fp8:
            aq = _QBUF[: T * INTER].view(T, INTER)
            H.silu_mul_quant(self.gu[:T], aq, _SBUF[:T])
            lin["down"].mm_q(aq, self.d[:T])
        else:
            H.silu_mul(self.gu[:T], self.act[:T])
            lin["down"](self.act[:T], self.d[:T])

    @torch.inference_mode()
    def chunk(self, ids, p0, st):
        """ids (T,) long on the device at positions p0..; returns the final-normed hidden (T, HIDDEN) view."""
        w, T = self.w, ids.shape[0]
        self.pos.fill_(p0)
        self.nrow.fill_(T)
        torch.index_select(w.embed, 0, ids, out=self.x[:T])
        ai = gi = 0
        for i, L in enumerate(w.layers):
            lin = self.lin[i]
            d = None if i == 0 else self.d[:T]
            if IS_ATTN[i]:
                self._attn(L, lin, ai, st, p0, T, d)
                ai += 1
            else:
                proj = self.proj[:T]
                lin["in_proj"].norm_in(self.x[:T], d, L["in_norm"], self.h[:T], proj)
                y = self.y[:T]
                H.gdn_conv(proj, st.conv[gi], L["conv"], L["A_neg"], L["dt_bias"], y, self.gb[:T])
                q = y[:, :2048].view(T, 16, 128)
                k = y[:, 2048:4096].view(T, 16, 128)
                v = y[:, 4096:].view(T, 48, 128)
                gdn_prefill(q, k, v, self.gb[:T, 0], self.gb[:T, 1], st.ssm[gi], out=self.o[:T])
                H.gated_norm(self.o[:T], proj, L["gnorm"], self.go[:T])
                H.conv_update(self.nrow, proj, st.conv[gi])
                lin["out"](self.go[:T], self.d[:T])
                gi += 1
            self._mlp(L, lin, T)
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
        self.mlin["fc"](self.mx[:T], self.x[:T])
        self._attn(M, self.mlin, 16, st, p0, T, None)
        self._mlp(M, self.mlin, T)
        out = torch.empty(1, HIDDEN, dtype=torch.bfloat16, device=hn.device)
        H.add_rmsnorm(self.x[T - 1:T], self.d[T - 1:T], M["norm"], out)
        return torch.matmul(out, w.lm_head.t()).float(), out
