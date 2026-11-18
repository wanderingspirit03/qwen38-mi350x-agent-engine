import time
import torch
import torch.nn.functional as F
from eng import ref_ops as R
from eng import hip as H

torch.manual_seed(0)
dev = "cuda"
PROJ = 16480


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
ssm = torch.randn(L, 48, 128, 128, device=dev) * 0.1          # [h, k, v]
ssmT = ssm.transpose(-1, -2).contiguous()                      # [h, v, k] for the HIP kernels
qk = torch.empty(L, T, 32, 128, device=dev)
v = torch.empty(L, T, 48, 128, device=dev)
gb = torch.empty(L, T, 2, 48, device=dev)
out = torch.empty(T, 6144, device=dev, dtype=torch.bfloat16)
obuf = torch.empty(16, 48, 128, device=dev)
for l in range(L):
    H.gdn_prep(proj[l], conv[l], cw[l], aneg[l], dtb[l], qk[l], v[l], gb[l])
    H.gdn_verify(qk[l], v[l], gb[l], ssmT[l], obuf, proj[l], gw[l], out)
    ro, rS, _ = ref_layer(proj[l], conv[l], cw[l], aneg[l], dtb[l], gw[l], ssm[l])
    print("verify layer", l, "max err", (out.float() - ro.float()).abs().max().item(), "ref max", ro.abs().max().item())
for n in (1, 3, 8):
    s2, c2 = ssmT.clone(), conv.clone()
    nd = torch.tensor([n], device=dev, dtype=torch.int32)
    H.gdn_commit(nd, qk, v, gb, s2, proj, c2)
    for l in range(L):
        _, rS, rc = ref_layer(proj[l, :n], conv[l], cw[l], aneg[l], dtb[l], gw[l], ssm[l])
        print("commit n", n, "layer", l, "state err", (s2[l].transpose(-1, -2) - rS).abs().max().item(),
              "conv err", (c2[l].float() - rc.float()).abs().max().item())
# timing
L = 48
proj = (torch.randn(L, T, PROJ, device=dev)).to(torch.bfloat16)
conv = torch.zeros(L, 3, 10240, device=dev, dtype=torch.bfloat16)
cw = torch.randn(L, 10240, 4, device=dev).to(torch.bfloat16)
aneg = -torch.rand(L, 48, device=dev)
dtb = torch.randn(L, 48, device=dev)
gw = torch.ones(L, 128, device=dev, dtype=torch.bfloat16)
ssm = torch.zeros(L, 48, 128, 128, device=dev)
qk = torch.empty(L, T, 32, 128, device=dev)
v = torch.empty(L, T, 48, 128, device=dev)
gb = torch.empty(L, T, 2, 48, device=dev)
nd = torch.tensor([5], device=dev, dtype=torch.int32)


def tm(f, n=100):
    for _ in range(5):
        f(0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        f(i)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


for TT in (1, 4, 8):
    a = tm(lambda i: H.gdn_prep(proj[i % 48, :TT], conv[i % 48], cw[i % 48], aneg[i % 48], dtb[i % 48], qk[i % 48],
                                v[i % 48], gb[i % 48]))
    b = tm(lambda i: H.gdn_verify(qk[i % 48], v[i % 48], gb[i % 48], ssm[i % 48], obuf, proj[i % 48, :TT], gw[i % 48], out))
    print(f"T={TT}: prep {a:.1f} us, verify {b:.1f} us per layer (includes launch)")
print(f"commit 48 layers n=5: {tm(lambda i: H.gdn_commit(nd, qk, v, gb, ssm, proj, conv), 20):.1f} us")
# in a graph (launch overhead removed)
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(48):
        H.gdn_prep(proj[i], conv[i], cw[i], aneg[i], dtb[i], qk[i], v[i], gb[i])
        H.gdn_verify(qk[i], v[i], gb[i], ssm[i], obuf, proj[i], gw[i], out)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    for i in range(48):
        H.gdn_prep(proj[i], conv[i], cw[i], aneg[i], dtb[i], qk[i], v[i], gb[i])
        H.gdn_verify(qk[i], v[i], gb[i], ssm[i], obuf, proj[i], gw[i], out)
print(f"graph of 48 x (prep+verify) T=8: {tm(lambda i: g.replay(), 50):.1f} us")

for name, fn in [("prep", lambda i: H.gdn_prep(proj[i], conv[i], cw[i], aneg[i], dtb[i], qk[i], v[i], gb[i])),
                 ("verify", lambda i: H.gdn_verify(qk[i], v[i], gb[i], ssm[i], obuf, proj[i], gw[i], out))]:
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        for i in range(48):
            fn(i)
    print(f"graph of 48 x {name} T=8: {tm(lambda i: g2.replay(), 50):.1f} us")

gc = torch.cuda.CUDAGraph()
with torch.cuda.graph(gc):
    H.gdn_commit(nd, qk, v, gb, ssm, proj, conv)
print(f"graph commit 48 layers n=5: {tm(lambda i: gc.replay(), 50):.1f} us")

for TT in (1, 2, 4, 8, 16):
    pj = torch.randn(TT, PROJ, device=dev).to(torch.bfloat16)
    qk2 = torch.empty(48, TT, 32, 128, device=dev); v2 = torch.empty(48, TT, 48, 128, device=dev)
    gb2 = torch.empty(48, TT, 2, 48, device=dev); o2 = torch.empty(TT, 6144, device=dev, dtype=torch.bfloat16)
    for i in range(48):
        H.gdn_prep(pj, conv[i], cw[i], aneg[i], dtb[i], qk2[i], v2[i], gb2[i])
    g3 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g3):
        for i in range(48):
            H.gdn_verify(qk2[i], v2[i], gb2[i], ssm[i], obuf, pj, gw[i], o2)
    print(f"graph 48 x verify T={TT}: {tm(lambda i: g3.replay(), 50):.1f} us")
x = torch.zeros(1, device=dev)
g4 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g4):
    for i in range(48):
        x.add_(1)
print(f"graph 48 x tiny torch kernel: {tm(lambda i: g4.replay(), 50):.1f} us")
