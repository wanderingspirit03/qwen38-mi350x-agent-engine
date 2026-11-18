import torch
from eng import hip as H
gu = (torch.randn(300, 2 * 17408, device="cuda") * 2).to(torch.bfloat16)
act = torch.empty(300, 17408, dtype=torch.bfloat16, device="cuda")
H.silu_mul(gu, act)
q1 = torch.empty(300, 17408, dtype=torch.float8_e4m3fn, device="cuda"); s1 = torch.empty(300, 1, device="cuda")
q2 = torch.empty_like(q1); s2 = torch.empty_like(s1)
H.quant_fp8(act, q1, s1)
H.silu_mul_quant(gu, q2, s2)
print("scale diff", (s1 - s2).abs().max().item(), "q mismatch", (q1.float() != q2.float()).float().mean().item())
