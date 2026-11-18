"""Prefix cache for one-sequence-at-a-time serving.

An entry is a past prompt: its token ids, the attention KV of every prompt position (16 target layers + MTP),
and GDN checkpoints (recurrent states and conv histories after the first c tokens, for a few c). A new prompt
resumes from the entry and checkpoint that let it skip the most tokens: c <= common prefix - 1 (the MTP layer's
KV at position c - 1 depends on token c) and c <= len - 1 (at least one token is computed for its logits).
"""
import time

import numpy as np
import torch


def lcp(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0
    d = np.flatnonzero(a[:n] != b[:n])
    return int(d[0]) if len(d) else n


class Entry:
    def __init__(self, tokens):
        self.tokens = tokens
        self.kv = None            # (2, 17, n, 4, 256) bf16
        self.ckpts = {}           # c -> (ssm (48, 48, 128, 128) fp32, conv (48, 3, 10240) bf16)
        self.used = time.monotonic()

    def nbytes(self):
        b = self.kv.numel() * 2 if self.kv is not None else 0
        for s, c in self.ckpts.values():
            b += s.numel() * 4 + c.numel() * 2
        return b


class PrefixCache:
    def __init__(self, budget_bytes, every=2048):
        self.entries = []
        self.budget = budget_bytes
        self.every = every
        self.in_work = None       # the entry whose prompt KV the working buffers hold

    def lookup(self, toks):
        """-> (entry, c, best_lcp): the resume point (entry None, c 0 when nothing matches)."""
        best, bc, bl = None, 0, 0
        n = len(toks)
        for e in self.entries:
            L = lcp(toks, e.tokens)
            bl = max(bl, L)
            lim = min(L - 1, n - 1)
            for c in e.ckpts:
                if bc < c <= lim:
                    best, bc = e, c
        return best, bc, bl

    def plan(self, n, c0, best_lcp):
        """Checkpoint positions for a prefill from c0: the divergence point and regular ones."""
        pts = set()
        if best_lcp - 1 > c0:
            pts.add(min(best_lcp - 1, n - 1))
        if best_lcp < self.every:
            # an unshared prompt: regular checkpoints let a later prompt resume near wherever it diverges. A prompt
            # that resumes from a shared prefix gets none: its tail is its own, and splitting costs prefill speed
            pts.update(range((c0 // self.every + 1) * self.every, n - 1, self.every))
        return sorted(p for p in pts if p > c0 + 64)        # no tiny chunks just for a checkpoint

    def restore(self, e, c, work_kv, ssm, conv):
        """Load entry e's state at c into the working buffers."""
        if self.in_work is not e:
            work_kv[:, :, :c].copy_(e.kv[:, :, :c])
        s, cv = e.ckpts[c]
        ssm.copy_(s)
        conv.copy_(cv)
        e.used = time.monotonic()

    def add(self, toks, work_kv, ckpts):
        """A new entry for prompt toks whose KV is in work_kv[:, :, :n]; ckpts {c: (ssm, conv)} already copied."""
        e = Entry(toks)
        n = len(toks)
        e.ckpts = ckpts
        self._make_room(e.nbytes() + work_kv[:, :, :n].numel() * 2)
        e.kv = work_kv[:, :, :n].clone()
        self.entries.append(e)
        self.in_work = e
        return e

    def _make_room(self, need):
        used = sum(e.nbytes() for e in self.entries)
        self.entries.sort(key=lambda e: e.used)
        while self.entries and used + need > self.budget:
            used -= self.entries.pop(0).nbytes()
