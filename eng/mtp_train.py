"""Self-distillation of the MTP draft. LoRA (rank R) on the MTP layer's five matrices plus deltas of its norm
vectors, trained on the engine's own greedy answers (eng.mtp_data shards in the development store, tests/store.sh)
with the chain objective of eng.draft_study.chains: depth 1 from the target's hidden, depth j >= 2 from the MTP's
own hidden of depth j - 1 (training-time test), loss = sum_j DECAY^(j-1) * cross-entropy of the draft logits (bf16
lm_head rows of the draft vocabulary + the prompt's extra tokens, as the engine) against the engine's token.
Evaluation: per-depth matches and the tokens per round of K = 9 rounds along the answers (as eng.draft_study).
python3 -m eng.mtp_train TRAIN_TAGS EVAL_TAGS EPOCHS OUT       (tags comma-separated; OUT.pt = the delta)"""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import store
from .main import MODEL
from .model import HEAD_DIM, HEADS, INTER, KV_HEADS, apply_rope, rope_cos_sin

SYS = 5957
EPS = 1e-6
RANK = int(os.environ.get("MT_RANK", "64"))
LR = float(os.environ.get("MT_LR", "2e-4"))
DEPTH = int(os.environ.get("MT_DEPTH", "6"))
DECAY = float(os.environ.get("MT_DECAY", "0.8"))
BATCH = int(os.environ.get("MT_BATCH", "4"))
EXTRA = 256
MATS = ("fc", "qkv", "o", "gate_up", "down")
NORMS = ("pre_emb", "pre_hid", "in_norm", "post_norm", "q_norm", "k_norm", "norm")


def load_weights():
    """embed, lm_head and the MTP layer (bf16, on the device), as eng.model.Weights names them"""
    import json
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    want = {"model.language_model.embed_tokens.weight": "embed", "lm_head.weight": "lm_head"}
    m = "mtp.layers.0."
    want.update({"mtp.fc.weight": "fc", "mtp.norm.weight": "norm", "mtp.pre_fc_norm_embedding.weight": "pre_emb",
                 "mtp.pre_fc_norm_hidden.weight": "pre_hid", m + "input_layernorm.weight": "in_norm",
                 m + "post_attention_layernorm.weight": "post_norm", m + "mlp.gate_proj.weight": "gate",
                 m + "mlp.up_proj.weight": "up", m + "mlp.down_proj.weight": "down",
                 m + "self_attn.q_proj.weight": "q", m + "self_attn.k_proj.weight": "k",
                 m + "self_attn.v_proj.weight": "v", m + "self_attn.o_proj.weight": "o",
                 m + "self_attn.q_norm.weight": "q_norm", m + "self_attn.k_norm.weight": "k_norm"})
    w = {}
    for name, short in want.items():
        with safe_open(os.path.join(MODEL, idx[name]), framework="pt", device="cpu") as sf:
            w[short] = sf.get_tensor(name).cuda()
    w["gate_up"] = torch.cat([w.pop("gate"), w.pop("up")])
    w["qkv"] = torch.cat([w.pop("q"), w.pop("k"), w.pop("v")])
    return w


class Mtp(torch.nn.Module):
    """the MTP layer with LoRA on its matrices (B = 0 at the start) and deltas of its norm vectors"""

    def __init__(self, w, rank):
        super().__init__()
        self.w = w
        g = torch.Generator(device="cuda").manual_seed(0)
        self.A = torch.nn.ParameterDict({k: torch.nn.Parameter(
            torch.randn(rank, w[k].shape[1], device="cuda", generator=g) / math.sqrt(w[k].shape[1])) for k in MATS})
        self.B = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.zeros(w[k].shape[0], rank, device="cuda"))
                                         for k in MATS})
        self.dn = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.zeros_like(w[k], dtype=torch.float32))
                                          for k in NORMS})
        self.on = True

    def lin(self, x, k):
        x = x.bfloat16()
        y = x @ self.w[k].t()
        if self.on:
            y = y + (x @ self.A[k].bfloat16().t()) @ self.B[k].bfloat16().t()
        return y.float()

    def rms(self, x, k=None):
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
        if k is None:
            return x
        nw = self.w[k].float() + (self.dn[k] if self.on else 0)
        return x * (1.0 + nw)

    def fc_rows(self, tok, hid):
        return self.lin(torch.cat([self.rms(self.w["embed"][tok], "pre_emb"), self.rms(hid, "pre_hid")], -1), "fc")

    def qkv_rows(self, x, pos):
        R = x.shape[0]
        qkv = self.lin(self.rms(x, "in_norm"), "qkv")
        qg = qkv[:, :HEADS * 512].view(R, HEADS, 512)
        q, gate = qg[..., :HEAD_DIM], qg[..., HEAD_DIM:].reshape(R, -1)
        k = qkv[:, HEADS * 512:HEADS * 512 + KV_HEADS * HEAD_DIM].view(R, KV_HEADS, HEAD_DIM)
        v = qkv[:, HEADS * 512 + KV_HEADS * HEAD_DIM:].view(R, KV_HEADS, HEAD_DIM)
        cos, sin = rope_cos_sin(pos, x.device)
        q = apply_rope(self.rms(q, "q_norm"), cos, sin)
        k = apply_rope(self.rms(k, "k_norm"), cos, sin)
        return q, gate, k, v

    def finish(self, x, q, gate, K1, V1, lim, Kc=None, Vc=None):
        a = attention(q, K1, V1, lim, Kc, Vc) * torch.sigmoid(gate)
        x = x + self.lin(a, "o")
        gu = self.lin(self.rms(x, "post_norm"), "gate_up")
        x = x + self.lin(F.silu(gu[:, :INTER]) * gu[:, INTER:], "down")
        return self.rms(x, "norm")

    def delta(self):
        """the trained delta, small: LoRA factors in fp16, norm deltas in fp32"""
        return {"A": {k: v.detach().half().cpu() for k, v in self.A.items()},
                "B": {k: v.detach().half().cpu() for k, v in self.B.items()},
                "dn": {k: v.detach().cpu() for k, v in self.dn.items()}}


def attention(q, K1, V1, lim, Kc=None, Vc=None):
    """as eng.draft_study.attention: row r sees real positions < lim[r] and its own chain keys Kc/Vc"""
    R, G = q.shape[0], HEADS // KV_HEADS
    qg = q.view(R, KV_HEADS, G, HEAD_DIM) * HEAD_DIM ** -0.5
    hi = int(lim.max())
    s = torch.einsum("rkgd,pkd->rkgp", qg.bfloat16(), K1[:hi].bfloat16()).float()
    s = s.masked_fill((torch.arange(hi, device=q.device)[None] >= lim[:, None])[:, None, None], float("-inf"))
    if Kc is not None:
        s = torch.cat([s, torch.einsum("rkgd,rmkd->rkgm", qg, Kc)], -1)
    p = s.softmax(-1)
    o = torch.einsum("rkgp,pkd->rkgd", p[..., :hi].bfloat16(), V1[:hi].bfloat16()).float()
    if Kc is not None:
        o = o + torch.einsum("rkgm,rmkd->rkgd", p[..., hi:], Vc)
    return o.reshape(R, HEADS * HEAD_DIM)


class Data:
    def __init__(self, tags):
        sy = store.load("sys.pt")
        self.hs = sy["hn"].cuda()
        self.seqs = []
        for tag in tags:
            k = 0
            while (sh := store.load(f"{tag}-{k}.pt")) is not None:
                for ids, n, hn in zip(sh["ids"], sh["n"], sh["hn"]):
                    if ids.shape[0] - 1 - n >= 2:
                        self.seqs.append((ids.long().cuda(), n, hn.cuda()))
                k += 1
        print(f"{','.join(tags)}: {len(self.seqs)} sequences, {sum(s[0].shape[0] - s[1] for s in self.seqs)} answer "
              f"tokens", flush=True)


DV = np.load(os.path.join(os.path.dirname(__file__), "draft_vocab.npy")).astype(np.int64)
IN_DV = np.zeros(248320, dtype=bool)
IN_DV[DV] = True


def sub_vocab(prompt):
    ids, cnt = np.unique(prompt[~IN_DV[prompt]], return_counts=True)
    ids = ids[np.argsort(-cnt, kind="stable")[:EXTRA]]
    return torch.from_numpy(np.concatenate([DV, ids])).cuda()


def run_seq(M, hs, S, n, hn, D, train, info=None):
    """the chain over all anchors of one sequence; returns the loss (train) and per-depth match bits (A, D);
    info (a list): gets (the target's top-1 probability, the draft token's rank in the target) per depth-1 row"""
    L = S.shape[0]
    HN = torch.cat([hs, hn])
    sub = sub_vocab(S[:n].cpu().numpy())
    pos_in = torch.full((248320,), -1, dtype=torch.long, device="cuda")
    pos_in[sub] = torch.arange(sub.shape[0], device="cuda")
    Hsub = M.w["lm_head"][sub]
    rows = torch.arange(L - 1, device="cuda")
    x = M.fc_rows(S[1:], HN[:L - 1])
    q, gate, K1, V1 = M.qkv_rows(x, rows)
    P = torch.arange(n, L - 1, device="cuda")
    A = P.shape[0]
    h = M.finish(x[P - 1], q[P - 1], gate[P - 1], K1, V1, P)
    m = torch.zeros(A, D, dtype=torch.bool, device="cuda")
    loss = 0.0
    Kc = Vc = None
    sel = torch.arange(A, device="cuda")
    for j in range(1, D + 1):
        if j > 1:
            keep = P + j <= L - 1
            if not keep.any():
                break
            P, h, sel = P[keep], h[keep], sel[keep]
            if Kc is not None:
                Kc, Vc = Kc[keep], Vc[keep]
            x = M.fc_rows(S[P + j - 1], h)
            q, gate, k, v = M.qkv_rows(x, P + j - 2)
            Kc = k[:, None] if Kc is None else torch.cat([Kc, k[:, None]], 1)
            Vc = v[:, None] if Vc is None else torch.cat([Vc, v[:, None]], 1)
            h = M.finish(x, q, gate, K1, V1, P, Kc, Vc)
        lg = (h.bfloat16() @ Hsub.t()).float()
        lab = pos_in[S[P + j]]
        m[sel, j - 1] = lg.argmax(-1) == lab
        if info is not None and j == 1:
            tl = (HN[P].bfloat16() @ M.w["lm_head"].t()).float()
            d = sub[lg.argmax(-1)]
            info.append(torch.stack([tl.softmax(-1).amax(-1), (tl > tl.gather(1, d[:, None])).sum(-1).float()], 1))
        if train:
            loss = loss + DECAY ** (j - 1) * F.cross_entropy(lg, lab, ignore_index=-1, reduction="sum")
    return loss, m


def tokens_per_round(ms, K=9):
    T = R = 0
    for m in ms:
        m = m.cpu().numpy()
        p = 0
        while p < m.shape[0]:
            a = 0
            while a < K and a < m.shape[1] and m[p, a]:
                a += 1
            T += a + 1
            R += 1
            p += a + 1
    return T / R


@torch.no_grad()
def evaluate(M, data, name, D=9):
    info = []
    ms = [run_seq(M, data.hs, S, n, hn, D, False, info)[1] for S, n, hn in data.seqs]
    info = torch.cat(info)
    m1 = torch.cat(ms)[:, 0]
    miss = info[~m1]
    conf = miss[:, 0]
    print(f"  depth-1 misses {miss.shape[0]} of {m1.shape[0]}: target top-1 prob < 0.5: {(conf < 0.5).float().mean():.2f}, "
          f"0.5-0.9: {((conf >= 0.5) & (conf < 0.9)).float().mean():.2f}, >= 0.9: {(conf >= 0.9).float().mean():.2f}; "
          f"draft token = target's 2nd choice: {(miss[:, 1] == 1).float().mean():.2f}", flush=True)
    if len(ms) == 96:
        print(f"  tokens/round first-turn (48) {tokens_per_round(ms[:48]):.3f}, later-turn (48) "
              f"{tokens_per_round(ms[48:]):.3f}", flush=True)
    allm = torch.cat(ms).float()
    cond = []
    reach = torch.ones(allm.shape[0], device="cuda", dtype=torch.bool)
    for j in range(D):
        c = allm[reach, j].mean().item() if reach.any() else float("nan")
        cond.append(round(c, 3))
        reach = reach & allm[:, j].bool()
    out = {"tpr": tokens_per_round(ms), "cond": cond}
    print(f"eval {name}: tokens/round K=9 {out['tpr']:.3f}; acceptance per depth given the earlier ones {cond}", flush=True)
    return out, ms


def quantized(M):
    """a copy of M with the delta merged and every matrix 4-bit round-to-nearest (eng.draft_study.fq4; the in/post
    norms folded into qkv/gate_up as the engine does) and a 4-bit head"""
    from .draft_study import fq4
    w = dict(M.w)
    on = M.on
    for k in MATS:
        Wk = w[k].float() + ((M.B[k] @ M.A[k]) if on else 0)
        nk = {"qkv": "in_norm", "gate_up": "post_norm"}.get(k)
        nw = None if nk is None else w[nk].float() + (M.dn[nk] if on else 0)
        w[k] = fq4(Wk, nw).bfloat16()
    for k in NORMS:
        w[k] = (w[k].float() + (M.dn[k] if on else 0)).bfloat16()
    w["in_norm"] = torch.zeros_like(w["in_norm"])
    w["post_norm"] = torch.zeros_like(w["post_norm"])
    w["lm_head"] = fq4(w["lm_head"]).bfloat16()
    Q = Mtp(w, 1)
    Q.on = False
    return Q


def main():
    tr_tags, ev_tags, epochs, out = sys.argv[1].split(","), sys.argv[2].split(","), float(sys.argv[3]), sys.argv[4]
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    w = load_weights()
    M = Mtp(w, RANK)
    ev = Data(ev_tags)
    tr = Data(tr_tags) if tr_tags != ["-"] else None
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    ck = store.load(out + "-ckpt.pt")
    if ck is not None:
        M.load_state_dict(ck["model"])
    M.on = False
    base, _ = evaluate(M, ev, "base")
    with torch.no_grad():
        evaluate(quantized(M), ev, "base, 4-bit")
    M.on = True
    if ck is not None:
        evaluate(M, ev, "checkpoint")
    if tr is None:
        return
    params = [p for p in M.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.99), weight_decay=0.0)
    steps = int(epochs * len(tr.seqs) / BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    step0 = 0
    if ck is not None and "opt" in ck:
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step0 = ck["step"]
    rng = np.random.default_rng(0)
    order = np.concatenate([rng.permutation(len(tr.seqs)) for _ in range(int(math.ceil(epochs)) + 1)])
    t1, tl, tn, tm = time.time(), 0.0, 0, None
    budget = float(os.environ.get("MT_MINUTES", "9")) * 60
    for s in range(step0, steps):
        opt.zero_grad(set_to_none=True)
        ntok = 0
        for i in order[s * BATCH:(s + 1) * BATCH]:
            S, n, hn = tr.seqs[i]
            ntok += S.shape[0] - 1 - n
        for i in order[s * BATCH:(s + 1) * BATCH]:
            S, n, hn = tr.seqs[i]
            loss, m = run_seq(M, tr.hs, S, n, hn, DEPTH, True)
            (loss / ntok).backward()
            tl += loss.item()
            tn += S.shape[0] - 1 - n
            mm = m.float().mean(0)
            tm = mm if tm is None else 0.98 * tm + 0.02 * mm
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if (s + 1) % 25 == 0:
            print(f"step {s + 1}/{steps} loss/token {tl / tn:.4f} train match per depth {[round(v, 3) for v in tm.tolist()]}"
                  f" {time.time() - t1:.0f} s", flush=True)
            tl = tn = 0
        if time.time() - t0 > budget and s + 1 < steps:
            store.save(out + "-ckpt.pt", {"model": M.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                                          "step": s + 1})
            print(f"checkpoint at step {s + 1}", flush=True)
            return
    res, ms = evaluate(M, ev, "trained")
    with torch.no_grad():
        evaluate(quantized(M), ev, "trained, 4-bit")
    print(f"tokens/round {base['tpr']:.3f} -> {res['tpr']:.3f} ({res['tpr'] / base['tpr'] - 1:+.1%})", flush=True)
    store.save(out + ".pt", M.delta())
    store.save(out + "-ckpt.pt", {"model": M.state_dict(), "step": steps})


if __name__ == "__main__":
    main()
