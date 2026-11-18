"""Qwen3.8-27B (Qwen3.5 architecture): weights and a forward pass over T consecutive positions of one sequence.
Version 0: plain PyTorch ops (eng.ref_ops), BF16 weights."""
import json
import os
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import ref_ops as R

HIDDEN, INTER, VOCAB = 5120, 17408, 248320
N_LAYERS = 64
HEADS, KV_HEADS, HEAD_DIM, ROT = 24, 4, 256, 64
K_HEADS, V_HEADS, GD = 16, 48, 128
QK_DIM, V_DIM = K_HEADS * GD, V_HEADS * GD          # 2048, 6144
CONV_DIM = 2 * QK_DIM + V_DIM                       # 10240
ROPE_THETA = 1e7
EPS = 1e-6
IS_ATTN = [(i % 4) == 3 for i in range(N_LAYERS)]
PRE = "model.language_model."


class Weights:
    def __init__(self, path, device="cuda"):
        t0 = time.time()
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        by_file = {}
        for name, f in idx.items():
            if name.startswith("model.visual."):
                continue
            by_file.setdefault(f, []).append(name)
        raw = {}
        for f, names in sorted(by_file.items()):
            with safe_open(os.path.join(path, f), framework="pt", device="cpu") as sf:
                for n in names:
                    raw[n] = sf.get_tensor(n).to(device, non_blocking=True)
        torch.cuda.synchronize()
        self.embed = raw.pop(PRE + "embed_tokens.weight")
        self.final_norm = raw.pop(PRE + "norm.weight")
        self.lm_head = raw.pop("lm_head.weight")
        self.layers = []
        for i in range(N_LAYERS):
            p = f"{PRE}layers.{i}."
            L = {"in_norm": raw.pop(p + "input_layernorm.weight"),
                 "post_norm": raw.pop(p + "post_attention_layernorm.weight"),
                 "gate_up": torch.cat([raw.pop(p + "mlp.gate_proj.weight"), raw.pop(p + "mlp.up_proj.weight")]),
                 "down": raw.pop(p + "mlp.down_proj.weight")}
            if IS_ATTN[i]:
                a = p + "self_attn."
                L["qkv"] = torch.cat([raw.pop(a + "q_proj.weight"), raw.pop(a + "k_proj.weight"),
                                      raw.pop(a + "v_proj.weight")])
                L["o"] = raw.pop(a + "o_proj.weight")
                L["q_norm"] = raw.pop(a + "q_norm.weight")
                L["k_norm"] = raw.pop(a + "k_norm.weight")
            else:
                a = p + "linear_attn."
                L["in_proj"] = torch.cat([raw.pop(a + "in_proj_qkv.weight"), raw.pop(a + "in_proj_z.weight"),
                                          raw.pop(a + "in_proj_b.weight"), raw.pop(a + "in_proj_a.weight")])
                L["conv"] = raw.pop(a + "conv1d.weight").reshape(CONV_DIM, 4)
                L["A_neg"] = -raw.pop(a + "A_log").float().exp()
                L["dt_bias"] = raw.pop(a + "dt_bias").float()
                L["gnorm"] = raw.pop(a + "norm.weight")
                L["out"] = raw.pop(a + "out_proj.weight")
            self.layers.append(L)
        m = "mtp.layers.0."
        self.mtp = {"fc": raw.pop("mtp.fc.weight"), "norm": raw.pop("mtp.norm.weight"),
                    "pre_emb": raw.pop("mtp.pre_fc_norm_embedding.weight"),
                    "pre_hid": raw.pop("mtp.pre_fc_norm_hidden.weight"),
                    "in_norm": raw.pop(m + "input_layernorm.weight"),
                    "post_norm": raw.pop(m + "post_attention_layernorm.weight"),
                    "gate_up": torch.cat([raw.pop(m + "mlp.gate_proj.weight"), raw.pop(m + "mlp.up_proj.weight")]),
                    "down": raw.pop(m + "mlp.down_proj.weight"),
                    "qkv": torch.cat([raw.pop(m + "self_attn.q_proj.weight"), raw.pop(m + "self_attn.k_proj.weight"),
                                      raw.pop(m + "self_attn.v_proj.weight")]),
                    "o": raw.pop(m + "self_attn.o_proj.weight"),
                    "q_norm": raw.pop(m + "self_attn.q_norm.weight"),
                    "k_norm": raw.pop(m + "self_attn.k_norm.weight")}
        assert not raw, list(raw)[:10]
        print(f"weights loaded in {time.time() - t0:.1f} s", flush=True)


class State:
    """One sequence: attention KV per attention layer, GDN conv history and recurrent state per GDN layer."""

    def __init__(self, max_len, device="cuda"):
        n_attn = sum(IS_ATTN) + 1          # the last one is the MTP layer's
        n_gdn = N_LAYERS - sum(IS_ATTN)
        self.k = torch.zeros(n_attn, max_len, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        self.v = torch.zeros_like(self.k)
        self.conv = torch.zeros(n_gdn, 3, CONV_DIM, dtype=torch.bfloat16, device=device)
        self.ssm = torch.zeros(n_gdn, V_HEADS, GD, GD, dtype=torch.float32, device=device)
        self.max_len = max_len

    def reset(self):
        self.conv.zero_()
        self.ssm.zero_()


def rope_cos_sin(pos, device):
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, ROT, 2, dtype=torch.float32, device=device) / ROT))
    f = pos.float()[:, None] * inv[None]
    emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """x (T, H, 256): rotate the first 64 dims (rotate-half form)."""
    xr = x[..., :ROT].float()
    x1, x2 = xr[..., :ROT // 2], xr[..., ROT // 2:]
    rot = torch.cat([-x2, x1], -1)
    out = xr * cos[:, None] + rot * sin[:, None]
    return torch.cat([out.to(x.dtype), x[..., ROT:]], -1)


class Model:
    def __init__(self, path, device="cuda"):
        self.w = Weights(path, device)
        self.device = device

    def gdn(self, L, li, x, st):
        T = x.shape[0]
        proj = F.linear(x, L["in_proj"])
        qkv, z, b, a = proj.split([CONV_DIM, V_DIM, V_HEADS, V_HEADS], -1)
        hist = torch.cat([st.conv[li], qkv], 0)                       # (T + 3, C)
        st.conv[li] = hist[-3:]
        w = L["conv"].float()
        y = sum(hist[j:j + T].float() * w[:, j] for j in range(4))
        y = F.silu(y)
        q, k, v = y.split([QK_DIM, QK_DIM, V_DIM], -1)
        q = R.l2norm(q.reshape(T, K_HEADS, GD)).repeat_interleave(3, 1) * GD ** -0.5
        k = R.l2norm(k.reshape(T, K_HEADS, GD)).repeat_interleave(3, 1)
        v = v.to(torch.bfloat16).float().reshape(T, V_HEADS, GD)
        beta = torch.sigmoid(b.float())
        g = L["A_neg"] * F.softplus(a.float() + L["dt_bias"])
        if T <= 16:
            o = R.gdn_recurrent(q, k, v, g, beta, st.ssm[li])
        else:
            o = R.gdn_chunked(q, k, v, g, beta, st.ssm[li])
        o = R.rmsnorm_gated(o.to(torch.bfloat16), z.reshape(T, V_HEADS, GD), L["gnorm"])
        return F.linear(o.reshape(T, V_DIM), L["out"])

    def attn(self, L, li, x, st, p0, cos, sin):
        T = x.shape[0]
        qkv = F.linear(x, L["qkv"])
        qg, k, v = qkv.split([HEADS * HEAD_DIM * 2, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], -1)
        qg = qg.reshape(T, HEADS, 2 * HEAD_DIM)
        q, gate = qg[..., :HEAD_DIM], qg[..., HEAD_DIM:]
        q = apply_rope(R.rmsnorm(q, L["q_norm"]), cos, sin)
        k = apply_rope(R.rmsnorm(k.reshape(T, KV_HEADS, HEAD_DIM), L["k_norm"]), cos, sin)
        st.k[li, p0:p0 + T] = k
        st.v[li, p0:p0 + T] = v.reshape(T, KV_HEADS, HEAD_DIM)
        n = p0 + T
        K = st.k[li, :n].transpose(0, 1).repeat_interleave(HEADS // KV_HEADS, 0)   # (24, n, 256)
        V = st.v[li, :n].transpose(0, 1).repeat_interleave(HEADS // KV_HEADS, 0)
        qi = torch.arange(p0, n, device=x.device)[:, None]
        kj = torch.arange(n, device=x.device)[None]
        o = F.scaled_dot_product_attention(q.transpose(0, 1), K, V, attn_mask=kj <= qi)   # (24, T, 256)
        o = o.transpose(0, 1).reshape(T, HEADS * HEAD_DIM) * torch.sigmoid(gate.reshape(T, -1).float()).to(o.dtype)
        return F.linear(o, L["o"])

    @torch.inference_mode()
    def forward(self, tokens, p0, st, last_only=True):
        """tokens: (T,) int64 on the device, at positions p0 .. p0+T-1. Returns logits (fp32) of the last
        position (or of all) and the final hidden states (after the final norm)."""
        W = self.w
        T = tokens.shape[0]
        x = W.embed[tokens]
        pos = torch.arange(p0, p0 + T, device=self.device)
        cos, sin = rope_cos_sin(pos, self.device)
        ai = gi = 0
        for i, L in enumerate(W.layers):
            h = R.rmsnorm(x, L["in_norm"])
            if IS_ATTN[i]:
                x = x + self.attn(L, ai, h, st, p0, cos, sin)
                ai += 1
            else:
                x = x + self.gdn(L, gi, h, st)
                gi += 1
            h = R.rmsnorm(x, L["post_norm"])
            gu = F.linear(h, L["gate_up"])
            x = x + F.linear(F.silu(gu[:, :INTER]) * gu[:, INTER:], L["down"])
        hn = R.rmsnorm(x, W.final_norm)
        sel = hn[-1:] if last_only else hn
        return F.linear(sel, W.lm_head).float(), hn

    @torch.inference_mode()
    def mtp_forward(self, tokens, hidden, p0, st):
        """The MTP layer: row i combines the target's (final-normed) hidden state at position p0+i with the
        embedding of the token at position p0+i+1 and predicts the token at p0+i+2. Returns logits of all rows
        and the MTP's own normed hidden states (the input for a chained step)."""
        W, M = self.w, self.w.mtp
        T = tokens.shape[0]
        e = R.rmsnorm(W.embed[tokens], M["pre_emb"])
        h = R.rmsnorm(hidden, M["pre_hid"])
        x = F.linear(torch.cat([e, h], -1), M["fc"])
        pos = torch.arange(p0, p0 + T, device=self.device)
        cos, sin = rope_cos_sin(pos, self.device)
        x = x + self.attn(M, st.k.shape[0] - 1, R.rmsnorm(x, M["in_norm"]), st, p0, cos, sin)
        h2 = R.rmsnorm(x, M["post_norm"])
        gu = F.linear(h2, M["gate_up"])
        x = x + F.linear(F.silu(gu[:, :INTER]) * gu[:, INTER:], M["down"])
        hn = R.rmsnorm(x, M["norm"])
        return F.linear(hn, W.lm_head).float(), hn
