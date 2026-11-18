"""Request handling: prefix-cache lookup, chunked prefill (with GDN checkpoints), MTP speculative decoding."""
import os
import time

import numpy as np
import torch

from .cache import PrefixCache
from .fast import Fast, SeqState
from .model import Weights
from .prefill import CHUNK, Prefill

STOP_IDS = (248046, 248044)
SPEC_K = int(os.environ.get("ENG_SPEC_K", "4"))
MAX_LEN = int(os.environ.get("ENG_MAX_LEN", "24576"))
CACHE_GB = float(os.environ.get("ENG_CACHE_GB", "80"))


class Engine:
    def __init__(self, path):
        self.w = Weights(path)
        self.st = SeqState(MAX_LEN)
        self.kv = torch.stack([self.st.k, self.st.v])      # (2, 17, MAX_LEN, 4, 256), the working KV
        self.st.k, self.st.v = self.kv[0], self.kv[1]
        self.pre = Prefill(self.w)
        self.fast = Fast(self.w, SPEC_K)
        self.cache = PrefixCache(int(CACHE_GB * 2**30))
        self.log = os.environ.get("ENG_LOG", "1") == "1"

    @staticmethod
    def _done(out, max_tokens):
        for i, t in enumerate(out):
            if t in STOP_IDS:
                return out[:i + 1], "stop"
        if len(out) >= max_tokens:
            return out[:max_tokens], "length"
        return None

    @torch.inference_mode()
    def generate(self, prompt, max_tokens):
        t0 = time.monotonic()
        st, pre = self.st, self.pre
        n = len(prompt)
        if n + max_tokens + SPEC_K + 2 > MAX_LEN:
            raise ValueError(f"prompt of {n} tokens is longer than this engine's limit")
        toks = np.asarray(prompt, dtype=np.int64)
        ids = torch.from_numpy(toks).cuda()
        e, c0, best = self.cache.lookup(toks)
        if e is not None:
            self.cache.restore(e, c0, self.kv, st.ssm, st.conv)
            if self.cache.in_work is not e:
                self.cache.in_work = None
        else:
            st.ssm.zero_()
            st.conv.zero_()
            self.cache.in_work = None
        pts = self.cache.plan(n, c0, best)
        bounds = sorted(set(pts) | {n} | set(range(c0 + CHUNK, n, CHUNK)))
        ckpts = {}
        s = c0
        for b in bounds:
            if b <= s:
                continue
            hn = pre.chunk(ids[s:b], s, st)
            if b < n:
                pre.mtp(ids[s + 1:b + 1], hn, s, st)
            if b in pts:
                ckpts[b] = (st.ssm.clone(), st.conv.clone())
            last = (s, b, hn)
            s = b
        s, b, hn = last
        x = int(pre.logits_last(hn).argmax())
        nxt = torch.cat([ids[s + 1:n], torch.tensor([x], device="cuda")])
        ml, mh = pre.mtp(nxt, hn, s, st)
        self.cache.add(toks, self.kv, ckpts)
        t1 = time.monotonic()
        out, rounds = self._decode(x, n, ml, mh, max_tokens)
        r = self._done(out, max_tokens)
        if self.log:
            t2 = time.monotonic()
            print(f"gen: prompt {n} resumed at {c0} (common {best}); prefill {1e3 * (t1 - t0):.0f} ms, "
                  f"decode {len(r[0])} tokens in {rounds} rounds {1e3 * (t2 - t1):.0f} ms", flush=True)
        return r

    def _decode(self, x, n, ml, mh, max_tokens):
        F, st = self.fast, self.st
        F.pos.fill_(n)
        F.vt[0] = x
        F.mhn[:1].copy_(mh)
        F.draft_chain(st, ml.argmax(-1))
        out = [x]
        rounds = 0
        while not self._done(out, max_tokens):
            F.round(st)
            k = int(F.n_acc.item())
            out += F.new[:k].tolist()
            rounds += 1
        return out, rounds
