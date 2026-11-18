import time
import torch
import torch.nn.functional as F
from eng import ref_ops as R
from eng.kernels.gdn_step import gdn_verify, gdn_commit, PROJ

torch.manual_seed(0)
dev = "cuda"


def ref_layer(proj, conv, cw, aneg, dtb, gw, ssm):
    T = proj.shape[0]
    qkv, z, b, a = proj.split([10240, 6144, 48, 48], -1)
    hist = torch.cat([conv, qkv], 0)
    y = sum(hist[j:j + T].float() * cw.float()[:, j] for j in range(4))
    y = F.silu(y).to(torch.bfloat16)
    q, k, v = y.split([2048, 2048, 6144], -1)
    q = R.l2norm(q.reshape(T, 16, 128)).repeat_interleave(3, 1) * 128 ** -0.5
    k = R.l2norm(k.reshape(T, 16, 128)).repeat_interleave(3, 1)
    v = v.float().reshape(T, 48, 128)
    beta = torch.sigmoid(b.float())
    g = aneg * F.softplus(a.float() + dtb)
    S = ssm.clone()
    o = R.gdn_recurrent(q, k, v, g, beta, S)
    o = R.rmsnorm_gated(o.to(torch.bfloat16), z.reshape(T, 48, 128), gw)
    return o.reshape(T, 6144), S, hist[-3:]


L, T = 3, 8
proj = (torch.randn(L, T, PROJ, device=dev) * 2).to(torch.bfloat16)
conv = (torch.randn(L, 3, 10240, device=dev) * 2).to(torch.bfloat16)
cw = (torch.randn(L, 10240, 4, device=dev) * 0.5).to(torch.bfloat16)
aneg = -torch.rand(L, 48, device=dev) * 2
dtb = torch.randn(L, 48, device=dev)
gw = (torch.rand(L, 128, device=dev) + 0.5).to(torch.bfloat16)
ssm = torch.randn(L, 48, 128, 128, device=dev) * 0.1
out = torch.empty(T, 6144, device=dev, dtype=torch.bfloat16)
for l in range(L):
    gdn_verify(proj[l], conv[l], cw[l], aneg[l], dtb[l], gw[l], ssm[l], out)
    ro, rS, _ = ref_layer(proj[l], conv[l], cw[l], aneg[l], dtb[l], gw[l], ssm[l])
    print("verify layer", l, "max err", (out.float() - ro.float()).abs().max().item(), "ref max", ro.abs().max().item())
for n in (1, 3, 8):
    ssm2, conv2 = ssm.clone(), conv.clone()
    nd = torch.tensor([n], device=dev, dtype=torch.int32)
    gdn_commit(proj, conv2, cw, aneg, dtb, ssm2, nd)
    for l in range(L):
        _, rS, rc = ref_layer(proj[l, :n], conv[l], cw[l], aneg[l], dtb[l], gw[l], ssm[l])
        print("commit n", n, "layer", l, "state err", (ssm2[l] - rS).abs().max().item(),
              "conv err", (conv2[l].float() - rc.float()).abs().max().item())
# timing
L = 48
proj = (torch.randn(L, T, PROJ, device=dev)).to(torch.bfloat16)
conv = torch.zeros(L, 3, 10240, device=dev, dtype=torch.bfloat16)
cw = torch.randn(L, 10240, 4, device=dev).to(torch.bfloat16)
aneg = -torch.rand(L, 48, device=dev)
dtb = torch.randn(L, 48, device=dev)
gw = torch.ones(L, 128, device=dev, dtype=torch.bfloat16)
ssm = torch.zeros(L, 48, 128, 128, device=dev)
nd = torch.tensor([5], device=dev, dtype=torch.int32)
for TT in (1, 4, 8):
    for _ in range(3):
        gdn_verify(proj[0, :TT], conv[0], cw[0], aneg[0], dtb[0], gw[0], ssm[0], out[:TT])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(100):
        gdn_verify(proj[i % 48, :TT], conv[i % 48], cw[i % 48], aneg[i % 48], dtb[i % 48], gw[i % 48], ssm[i % 48], out[:TT])
    torch.cuda.synchronize()
    print(f"verify T={TT}: {(time.perf_counter() - t0) / 100 * 1e6:.1f} us/layer")
for _ in range(3):
    gdn_commit(proj, conv, cw, aneg, dtb, ssm, nd)
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(20):
    gdn_commit(proj, conv, cw, aneg, dtb, ssm, nd)
torch.cuda.synchronize()
print(f"commit all 48 layers n=5: {(time.perf_counter() - t0) / 20 * 1e6:.1f} us")
