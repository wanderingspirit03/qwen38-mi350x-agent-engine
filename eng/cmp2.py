import torch, torch.nn.functional as F
from eng import model as M, ref_ops as R, hip as H
from eng.prefill import Prefill, gdn_chunk_ref
from eng.fast import SeqState
torch.manual_seed(0)
w = M.Weights("/models/Qwen3.8-27B")
L = w.layers[0]
T = 64
ids = torch.randint(0, 200000, (T,), device="cuda")
x = w.embed[ids]
h = R.rmsnorm(x, L["in_norm"])
h2 = torch.empty_like(x); xx = x.clone()
H.add_rmsnorm(xx, None, L["in_norm"], h2)
print("norm", (h.float() - h2.float()).abs().max().item())
proj = F.linear(h, L["in_proj"])
qkv, z, b, a = proj.split([10240, 6144, 48, 48], -1)
y = F.silu(sum(torch.cat([torch.zeros(3, 10240, device="cuda", dtype=torch.bfloat16), qkv])[j:j + T].float() * L["conv"].float()[:, j] for j in range(4)))
q, k, v = y.split([2048, 2048, 6144], -1)
qn = R.l2norm(q.reshape(T, 16, 128)) * 128 ** -0.5
kn = R.l2norm(k.reshape(T, 16, 128))
g = L["A_neg"] * F.softplus(a.float() + L["dt_bias"]); beta = torch.sigmoid(b.float())
qk = torch.empty(T, 32, 128, device="cuda"); vv = torch.empty(T, 48, 128, device="cuda"); gb = torch.empty(T, 2, 48, device="cuda")
H.gdn_prep(proj, torch.zeros(3, 10240, device="cuda", dtype=torch.bfloat16), L["conv"], L["A_neg"], L["dt_bias"], qk, vv, gb)
print("q", (qk[:, :16] - qn).abs().max().item(), "k", (qk[:, 16:] - kn).abs().max().item(), "v", (vv - v.reshape(T, 48, 128)).abs().max().item(), "|v|", v.abs().max().item())
print("g", (gb[:, 0] - g).abs().max().item(), "beta", (gb[:, 1] - beta).abs().max().item())
S0 = torch.zeros(48, 128, 128, device="cuda")
o_ref = R.gdn_recurrent(qn.repeat_interleave(3, 1), kn.repeat_interleave(3, 1), v.reshape(T, 48, 128), g, beta, S0.clone())
sT = torch.zeros(48, 128, 128, device="cuda")
o1 = gdn_chunk_ref(qk, vv, gb, sT)
print("o", (o1 - o_ref).abs().max().item(), "|o|", o_ref.abs().max().item(), o1.shape, o1.is_contiguous())
go_ref = R.rmsnorm_gated(o_ref.to(torch.bfloat16), z.reshape(T, 48, 128), L["gnorm"]).reshape(T, 6144)
go = torch.empty(T, 6144, device="cuda", dtype=torch.bfloat16)
H.gated_norm(o1.contiguous(), proj, L["gnorm"], go)
print("gated", (go.float() - go_ref.float()).abs().max().item(), "|go|", go_ref.abs().max().item())
