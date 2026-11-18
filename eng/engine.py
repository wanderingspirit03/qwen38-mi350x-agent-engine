"""The generation loop (version 0: chunked prefill, then one token per forward pass)."""
import time

import torch

from .model import Model, State

STOP_IDS = (248046, 248044)
PREFILL_CHUNK = 4096


class Engine:
    def __init__(self, path, max_len=20480):
        self.model = Model(path)
        self.st = State(max_len)

    @torch.inference_mode()
    def generate(self, prompt, max_tokens):
        st, m = self.st, self.model
        st.reset()
        ids = torch.tensor(prompt, dtype=torch.long, device="cuda")
        n = len(prompt)
        for p0 in range(0, n, PREFILL_CHUNK):
            logits, _ = m.forward(ids[p0:p0 + PREFILL_CHUNK], p0, st)
        out = []
        tok = int(logits[-1].argmax())
        while True:
            out.append(tok)
            if tok in STOP_IDS:
                return out, "stop"
            if len(out) >= max_tokens or n + len(out) >= st.max_len:
                return out, "length"
            logits, _ = m.forward(torch.tensor([tok], device="cuda"), n + len(out) - 1, st)
            tok = int(logits[-1].argmax())
