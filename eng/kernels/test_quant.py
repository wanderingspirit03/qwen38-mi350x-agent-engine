import torch
from eng import hip as H
x = (torch.randn(300, 5120, device="cuda") * 3).to(torch.bfloat16)
x[5, 7] = 100
q = torch.empty(300, 5120, dtype=torch.float8_e4m3fn, device="cuda"); s = torch.empty(300, 1, device="cuda")
H.quant_fp8(x, q, s)
rs = x.float().abs().amax(1, keepdim=True) / 448
rq = (x.float() / rs).to(torch.float8_e4m3fn)
print("scale err", (s - rs).abs().max().item(), "q mismatch", (q.float() != rq.float()).float().mean().item(),
      "deq err", ((q.float() * s) - x.float()).abs().max().item())
