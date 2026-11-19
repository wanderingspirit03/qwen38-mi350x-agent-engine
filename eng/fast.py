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
from .gptq import gptq_codes
from .kernels import w4 as w4q
from .kernels import w4a8
from .kernels.attention import attn_decode
from .model import (CONV_DIM, HEAD_DIM, HEADS, HIDDEN, INTER, IS_ATTN, KV_HEADS, N_LAYERS, V_DIM, VOCAB)

PROJ = CONV_DIM + V_DIM + 96
GDN_FUSED = os.environ.get("ENG_GDN_FUSED", "1") == "1"   # gdn_fused (one kernel per GDN layer) in verify


class SeqState:
    """The decode-side state of one sequence: attention KV (16 target layers + the MTP layer), GDN recurrent
    states in [layer, head, v, k] layout, conv histories."""

    def __init__(self, max_len, dev="cuda"):
        self.k = torch.zeros(17, max_len, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        self.v = torch.zeros_like(self.k)
        self.ssm = torch.zeros(48, 48, 128, 128, dtype=torch.float32, device=dev)
        self.conv = torch.zeros(48, 3, CONV_DIM, dtype=torch.bfloat16, device=dev)
        self.max_len = max_len


HEAD_BITS = int(os.environ.get("ENG_HEAD_BITS", "8"))     # target LM head: 4, 8 or 16 bits (the draft head is 4-bit)
W4A8 = os.environ.get("ENG_W4A8", "1")    # 1: int8-activation GEMMs (eng.kernels.w4a8) in the target model; mtp: also MTP
A8_ACC = os.environ.get("ENG_A8_ACC", "1") == "1"     # residual-adding W4A8 GEMMs split K, sums added by the next quant


EXTRA = int(os.environ.get("ENG_DRAFT_EXTRA", "256"))   # draft-vocabulary slots for prompt tokens (multiple of 16)
FOLD_NORM = os.environ.get("ENG_FOLD_NORM", "1") == "1"
# lossy acceptance (see accept_kernel): accept a draft whose target probability is >= tau times the top one
# (0: exact argmax matches only) and, if ENG_LOSSY_PRICE > 0, whose logit margin is <= price * the tokens it gains
LOSSY_TAU = float(os.environ.get("ENG_LOSSY_TAU", "0.3"))
LOSSY_PRICE = float(os.environ.get("ENG_LOSSY_PRICE", "0"))


def lossy_params(tau, price=0.0):
    """[price, cap] for accept_kernel (price 0 there: off)."""
    if tau <= 0:
        return [0.0, 0.0]
    return [price if price > 0 else 1e9, float(np.log(1 / tau))]


if LOSSY_TAU > 0:
    print(f"lossy acceptance: tau {LOSSY_TAU}, price {LOSSY_PRICE}", flush=True)
_XN = torch.empty(16, 5120, dtype=torch.bfloat16, device="cuda")


class Linear:
    """x (M <= 16, K) @ W^T on 4-bit weights (eng.kernels.w4). norm: the RMSNorm weight in front of W, folded in
    (x is then the raw residual stream); swiglu: W is [gate; up] and the output is silu(gate) * up;
    add=True: out += x @ W^T (out is the residual stream)."""

    def __init__(self, w, norm=None, swiglu=False, H=None):
        # folding the norm weight into the 4-bit W roughly doubles its error (local gap 0.0077 vs 0.0033 with
        # ENG_FOLD_NORM=0), but the separate norm kernel costs ~15% decode
        # H: X^T X of the input (x / rms(x) when the norm is folded) for GPTQ (eng.gptq)
        self.norm = norm if norm is not None and not FOLD_NORM else None
        if self.norm is not None:
            H = None
        self.p = w4q.quantize(w, norm_weight=None if self.norm is not None else norm, swiglu=swiglu, H=H)

    @staticmethod
    def from_codes(u, s, z):
        """A Linear from 4-bit codes, scales and zero points (w4.rtn_search's output layout), no norm."""
        lin = Linear.__new__(Linear)
        lin.norm = None
        qw, meta = w4q._pack(u, s, z)
        w4q._workspace(qw.device)
        lin.p = {"qw": qw, "meta": meta, "N": u.shape[0], "K": u.shape[1] * u.shape[2], "G": u.shape[2],
                 "rms": False, "swiglu": False}
        return lin

    def __call__(self, x, out, add=False, argmax=False):
        if self.norm is not None:
            xn = _XN[:x.shape[0]]
            H.add_rmsnorm(x, None, self.norm, xn)
            x = xn
        w4q.gemm(x, self.p, out=out, residual=out if add else None, argmax=argmax)


class A8Linear:
    """As Linear, with int8 activations (eng.kernels.w4a8; the norm weight is applied to x, not folded into W).
    x is quantized by a small kernel in front of each GEMM, into buffers shared per (M, K).
    add=True (A8_ACC): the GEMM splits K over workgroups and adds fp32 sums into a buffer kept per residual
    tensor; the next A8Linear reading that tensor adds them in while quantizing it, else flush() does."""
    xa = {}
    acc = {}        # residual tensor's data_ptr -> fp32 (16, K)
    pending = {}    # data_ptr -> rows with sums not yet added

    def __init__(self, w, norm=None, swiglu=False, H=None):
        self.p = w4a8.quantize(w, norm_weight=norm, swiglu=swiglu, H=H)

    @staticmethod
    def _xa(x):
        key = (x.shape[0], x.shape[1], x.device)
        if key not in A8Linear.xa:
            A8Linear.xa[key] = w4a8.new_xa(x.shape[0], x.shape[1], x.device)
        return A8Linear.xa[key]

    @staticmethod
    def _take(x):
        rows = A8Linear.pending.pop(x.data_ptr(), None)
        assert rows is None or rows == x.shape[0]
        return None if rows is None else A8Linear.acc[x.data_ptr()][:rows]

    @staticmethod
    def flush(x):
        """Add pending GEMM sums to the residual tensor x."""
        acc = A8Linear._take(x)
        if acc is not None:
            w4a8.quant_x(x, A8Linear._xa(x), acc=acc)

    def __call__(self, x, out, add=False):
        """x: (M, K) bf16, or an already quantized activation dict"""
        if not isinstance(x, dict):
            x = w4a8.quant_x(x, A8Linear._xa(x), self.p["norm"], acc=A8Linear._take(x))
        if add and A8_ACC:
            k = out.data_ptr()
            if k not in A8Linear.acc:
                A8Linear.acc[k] = torch.zeros(16, out.shape[1], dtype=torch.float32, device=out.device)
            assert A8Linear.pending.get(k, out.shape[0]) == out.shape[0]
            A8Linear.pending[k] = out.shape[0]
            w4a8.gemm(x, self.p, acc=A8Linear.acc[k][:out.shape[0]])
        else:
            assert out.data_ptr() not in A8Linear.pending
            w4a8.gemm(x, self.p, out=out, residual=out if add else None)


class W8Linear(Linear):
    def __init__(self, w):
        self.norm = None
        self.p = w4q.quantize8(w)


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
    def __init__(self, w, K, dev="cuda", hess=None, thess=None):
        """hess: eng.gptq.mtp_hessians(...) for GPTQ of the MTP layer and the draft head; thess:
        eng.gptq.TargetHessians for GPTQ of the target's W4A8 weights (None: round-to-nearest)"""
        self.w, self.K, self.R = w, K, K + 1
        R = self.R
        bf = dict(dtype=torch.bfloat16, device=dev)
        self.lin = []
        for i, L in enumerate(w.layers):
            TL = A8Linear if W4A8 != "0" else Linear
            th = thess.layer(i) if thess is not None and TL is A8Linear else None
            self.lin.append(self._layer_lin(L, IS_ATTN[i], TL, H=th))
        M = w.mtp
        ML = A8Linear if W4A8 == "mtp" else Linear
        mh = hess if hess is not None and ML is Linear else {}
        self.mlin = self._layer_lin(M, True, ML, H=mh)
        self.mlin["fc"] = ML(M["fc"], H=mh.get("fc"))
        HL = {4: Linear, 8: W8Linear, 16: Bf16Linear}[HEAD_BITS]
        self.head = HL(w.lm_head)
        # drafts are chosen from the 32768 most frequent output tokens (eng/draft_vocab.npy): an 8x smaller head,
        # plus EXTRA slots for the prompt's tokens outside that list (set_extra; identifiers copied from the context)
        dv = np.load(os.path.join(os.path.dirname(__file__), "draft_vocab.npy"))
        self.dv_np = dv.astype(np.int64)
        self.in_dv = np.zeros(VOCAB, dtype=bool)
        self.in_dv[dv] = True
        dv = np.concatenate([dv, np.full(EXTRA, dv[0])])
        self.dvocab = torch.from_numpy(dv.astype(np.int64)).to(dev)
        if EXTRA:   # every lm_head row's 4-bit codes, scales and zero points (as quantize() makes them), for set_extra
            self.hu = torch.empty(VOCAB, HIDDEN // 128, 128, dtype=torch.uint8, device=dev)
            self.hs = torch.empty(VOCAB, HIDDEN // 128, dtype=torch.float16, device=dev)
            self.hz = torch.empty(VOCAB, HIDDEN // 128, dtype=torch.uint8, device=dev)
            for r0 in range(0, VOCAB, 8192):
                wc = w.lm_head[r0:r0 + 8192].float()
                if hess is None:
                    u, s, z = w4q.rtn_search(wc.reshape(-1, HIDDEN // 128, 128))
                else:
                    u, s, z = gptq_codes(wc, hess["head"])
                self.hu[r0:r0 + 8192], self.hs[r0:r0 + 8192], self.hz[r0:r0 + 8192] = u, s, z
            t = self.dvocab
            self.dhead = Linear.from_codes(self.hu[t].int(), self.hs[t].float(), self.hz[t].float())
        else:
            self.dhead = Linear(w.lm_head[self.dvocab].contiguous(), H=None if hess is None else hess["head"])
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
        self.goq = w4a8.new_xa(R, V_DIM, dev)
        self.aoq = {}                                     # rows -> quantized attention output (W4A8)
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
        self.am = w4q.argmax_slots(dev)
        self.mhn = torch.empty(R, HIDDEN, **bf)
        # round state on the device
        i32 = dict(dtype=torch.int32, device=dev)
        self.pos = torch.zeros(1, **i32)                  # position of x
        self.vt = torch.zeros(R, dtype=torch.long, device=dev)   # [x, d_1..d_K]
        self.y = torch.zeros(R, dtype=torch.long, device=dev)
        self.lossy = torch.tensor(lossy_params(LOSSY_TAU, LOSSY_PRICE) + [0.0, 0.0], device=dev)   # see accept_kernel
        self.n_acc = torch.ones(1, **i32)
        self.new = torch.zeros(R, dtype=torch.long, device=dev)  # the round's new tokens (first n_acc valid)
        self.pick = torch.zeros(1, dtype=torch.long, device=dev)
        # positions derived from pos (see accept_kernel): [catch-up pos, its keys, verify keys, (chain pos, keys)...]
        self.P = torch.zeros(3 + 2 * K, **i32)
        self.res = torch.zeros(R + 1, dtype=torch.long, device=dev)   # [n_acc, new tokens...]
        self.zero = torch.zeros(1, dtype=torch.long, device=dev)

    def set_extra(self, prompt):
        """Fill the draft vocabulary's EXTRA slots with the prompt's tokens outside draft_vocab.npy (the most
        frequent ones if there are more); unused slots repeat dv[0]. prompt: int64 numpy array. Eager."""
        if not EXTRA:
            return
        ids, cnt = np.unique(prompt[~self.in_dv[prompt]], return_counts=True)
        ids = ids[np.argsort(-cnt, kind="stable")[:EXTRA]]
        ids = np.concatenate([ids, np.full(EXTRA - len(ids), self.dv_np[0])])
        t = torch.from_numpy(ids).to(self.dvocab.device)
        qw, meta = w4q._pack(self.hu[t].int(), self.hs[t].float(), self.hz[t].float())
        n0 = len(self.dv_np) // 16
        for k, src in (("qw", qw), ("meta", meta)):
            self.dhead.p[k].view(self.dhead.p["N"] // 16, -1)[n0:].copy_(src.view(EXTRA // 16, -1))
        self.dvocab[len(self.dv_np):].copy_(t)

    # ---- the target model on R rows ----
    @staticmethod
    def _layer_lin(L, attn, Lin=Linear, H=None):
        H = H or {}             # X^T X per GEMM for GPTQ
        d = {"gate_up": Lin(L["gate_up"], L["post_norm"], swiglu=True, H=H.get("gate_up")),
             "down": Lin(L["down"], H=H.get("down"))}
        if attn:
            d["qkv"], d["o"] = Lin(L["qkv"], L["in_norm"], H=H.get("qkv")), Lin(L["o"], H=H.get("o"))
        else:
            d["in_proj"], d["out"] = Lin(L["in_proj"], L["in_norm"], H=H.get("in_proj")), Lin(L["out"], H=H.get("out"))
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
                if isinstance(lin["out"], A8Linear) and GDN_FUSED:   # one kernel; the output quantized for the GEMM
                    H.gdn_fused(proj, st.conv[gi], L["conv"], L["A_neg"], L["dt_bias"], st.ssm[gi], L["gnorm"],
                                self.qk[gi], self.vv[gi], self.gb[gi], self.goq)
                    lin["out"](self.goq, self.x, add=True)
                    gi += 1
                    self.mlp(lin, R)
                    continue
                H.gdn_prep(proj, st.conv[gi], L["conv"], L["A_neg"], L["dt_bias"], self.qk[gi], self.vv[gi],
                           self.gb[gi])
                if isinstance(lin["out"], A8Linear):     # the gated norm quantizes its output for the GEMM
                    H.gdn_verify_q(self.qk[gi], self.vv[gi], self.gb[gi], st.ssm[gi], self.ob, proj, L["gnorm"],
                                   self.goq)
                    lin["out"](self.goq, self.x, add=True)
                else:
                    H.gdn_verify(self.qk[gi], self.vv[gi], self.gb[gi], st.ssm[gi], self.ob, proj, L["gnorm"],
                                 self.go)
                    lin["out"](self.go, self.x, add=True)
                gi += 1
            self.mlp(lin, R)
        A8Linear.flush(self.x)
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
        xa = None
        if isinstance(lin["o"], A8Linear):     # the attention output is quantized for the GEMM by the merge kernel
            if R not in self.aoq:
                self.aoq[R] = w4a8.new_xa(R, HEADS * HEAD_DIM, self.x.device)
            xa = self.aoq[R]
        attn_decode(self.q[:R], st.k[li], st.v[li], nkv, gate=gate, out=self.ao[:R].view(R, HEADS, HEAD_DIM), xa=xa)
        lin["o"](self.ao[:R] if xa is None else xa, self.x[:R], add=True)

    # ---- acceptance and commit ----
    def accept(self, st):
        H.accept(self.vt, self.y, self.n_acc, self.new, self.res, self.pick, self.pos, self.P, self.logits, self.lossy)
        H.gdn_commit(self.n_acc, self.qk, self.vv, self.gb, st.ssm, self.proj, st.conv)

    # ---- MTP ----
    def mtp(self, st, tokens, hidden, pos, nkv, rows, pick, out, decode=True):
        """The MTP layer on `rows` rows at positions pos.. (row i: target-side hidden at pos+i and the token at
        pos+i+1). Writes the draft (argmax of row `pick`) to out (1,) and leaves that row's normed hidden
        in self.mhn[:1]. decode=False: the draft stays in the argmax slots self.am until the next 1-row mtp()
        with tokens = out takes it (saves a kernel)."""
        w, M, R = self.w, self.w.mtp, rows
        H.mtp_in(tokens, w.embed, hidden, M["pre_emb"], M["pre_hid"], self.mx[:R],
                 am=self.am if R == 1 else None, map=self.dvocab)
        self.mlin["fc"](self.mx[:R], self.x[:R])
        self.attn(self.mlin, M, 16, st, pos, nkv, R)
        self.mlp(self.mlin, R)
        A8Linear.flush(self.x[:R])
        if R == 1:
            H.add_rmsnorm(self.x[:1], None, M["norm"], self.mhn[:1])
        else:
            H.add_rmsnorm(self.x[:R], None, M["norm"], self.mh[:R])
            self.mhn[:1].copy_(self.mh[:R].index_select(0, pick))
        self.dhead(self.mhn[:1], self.mlogit, argmax=True)
        if decode:
            H.am_decode(self.am, out, self.dvocab)

    def draft_chain(self, st):
        """vt[1] holds d_1: fill vt[2:] with K - 1 chained MTP drafts (position of x is self.pos)."""
        for j in range(1, self.K):
            P = self.P[3 + 2 * (j - 1):]
            self.mtp(st, self.vt[j:j + 1], self.mhn[:1], P[0:1], P[1:2], 1, self.zero, self.vt[j + 1:j + 2],
                     decode=j == self.K - 1)

    def round(self, st):
        self.verify(st)
        self.accept(st)
        # catch-up: rows i = 0..K: hidden at pos+i, token new[i] at pos+i+1; row a = n_acc - 1 gives d_1
        self.mtp(st, self.new, self.hn, self.P[0:1], self.P[1:2], self.R, self.pick, self.vt[1:2], decode=self.K == 1)
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
