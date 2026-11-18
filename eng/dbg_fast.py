import torch
import eng.fast as FM
from eng.fast import Fast, SeqState
from eng.model import Weights
w = Weights("/models/Qwen3.8-27B")
st = SeqState(16384)
F = Fast(w, 8)
print("init ok", flush=True)
orig = FM.Linear.__call__
def call(self, x, out, add=False):
    print("gemm", tuple(x.shape), tuple(out.shape), self.p["N"], self.p["K"], self.p["rms"], self.p["swiglu"], add, flush=True)
    orig(self, x, out, add); torch.cuda.synchronize()
FM.Linear.__call__ = call
F.vt.fill_(100); F.pos.fill_(5000); F.mpos.fill_(5000)
for name, fn in [("verify", lambda: F.verify(st)), ("accept", lambda: F.accept(st)),
                 ("mtp", lambda: F.mtp(st, F.new, F.hn, F.pos, F.R, F.zero)), ("chain", lambda: F.draft_chain(st))]:
    fn(); torch.cuda.synchronize(); print(name, "ok", flush=True)
