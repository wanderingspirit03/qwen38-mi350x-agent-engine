"""attn_prep and gdn_conv: check against torch references and time them (prefill sizes).
python3 -m eng.kernels.test_prep"""
import torch

from eng import hip as H
from .test_w4 import graph_time

dev = "cuda"


def ref_prep(qkv, qw, kw, pos):
    R = qkv.shape[0]
    q = qkv[:, :12288].view(R, 24, 512)[..., :256].float()
    k = qkv[:, 12288:13312].view(R, 4, 256).float()
    v = qkv[:, 13312:].view(R, 4, 256)

    def nr(x, w):
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + w.float())
        x = x.bfloat16().float()
        f = torch.arange(32, device=dev)
        inv = torch.pow(torch.tensor(1e7, device=dev), -(2 * f).float() / 64)
        ang = (pos + torch.arange(R, device=dev))[:, None].float() * inv[None]
        c, s = ang.cos()[:, None], ang.sin()[:, None]
        a, b = x[..., :32].clone(), x[..., 32:64].clone()
        x[..., :32] = a * c - b * s
        x[..., 32:64] = b * c + a * s
        return x.bfloat16()
    return nr(q, qw), nr(k, kw), v


def check_prep(R, pos=6400):
    qkv = torch.randn(R, 14336, device=dev).bfloat16()
    qw, kw = (torch.randn(256, device=dev) * 0.2).bfloat16(), (torch.randn(256, device=dev) * 0.2).bfloat16()
    kc = torch.zeros(16384, 4, 256, dtype=torch.bfloat16, device=dev)
    vc = torch.zeros_like(kc)
    qo = torch.empty(R, 24, 256, dtype=torch.bfloat16, device=dev)
    pd = torch.tensor([pos], dtype=torch.int32, device=dev)
    H.attn_prep(qkv, qw, kw, pd, qo, kc, vc)
    rq, rk, rv = ref_prep(qkv, qw, kw, pos)
    eq = (qo.float() - rq.float()).abs().max().item()
    ek = (kc[pos:pos + R].float() - rk.float()).abs().max().item()
    ev = (vc[pos:pos + R] != rv).sum().item()
    nq = (qo != rq).float().mean().item()
    t = graph_time([lambda: H.attn_prep(qkv, qw, kw, pd, qo, kc, vc)] * 20) * 1e6
    mb = R * (6144 + 2048) * 2 * 2 / 1e6
    print(f"attn_prep R={R}: max err q {eq:.3g} k {ek:.3g}, v mismatches {ev}, q differing {nq:.4f}; "
          f"{t:.1f} us ({mb / t:.2f} TB/s)", flush=True)


def check_conv(T):
    proj = torch.randn(T, 16480, device=dev).bfloat16()
    hist = torch.randn(3, 10240, device=dev).bfloat16()
    cw = (torch.randn(10240, 4, device=dev) * 0.3).bfloat16()
    aneg = -torch.rand(48, device=dev)
    dtb = torch.randn(48, device=dev)
    y = torch.empty(T, 10240, dtype=torch.bfloat16, device=dev)
    gb = torch.empty(T, 2, 48, device=dev)
    H.gdn_conv(proj, hist, cw, aneg, dtb, y, gb)
    xx = torch.cat([hist, proj[:, :10240]]).float()
    acc = sum(xx[j:j + T] * cw[:, j].float() for j in range(4))
    ref = (acc * torch.sigmoid(acc)).bfloat16()
    e = (y.float() - ref.float()).abs().max().item()
    nd = (y != ref).float().mean().item()
    a = proj[:, 10240 + 6144 + 48:].float()
    b = proj[:, 10240 + 6144:10240 + 6144 + 48].float()
    eg = (gb[:, 0] - aneg * torch.nn.functional.softplus(a + dtb)).abs().max().item()
    eb = (gb[:, 1] - torch.sigmoid(b)).abs().max().item()
    t = graph_time([lambda: H.gdn_conv(proj, hist, cw, aneg, dtb, y, gb)] * 20) * 1e6
    mb = T * 10240 * 2 * 2 / 1e6
    print(f"gdn_conv T={T}: max err {e:.3g} (differing {nd:.4f}), g {eg:.2g} beta {eb:.2g}; {t:.1f} us "
          f"({mb / t:.2f} TB/s)", flush=True)


if __name__ == "__main__":
    for R in (10, 1000, 5600):
        check_prep(R)
    for T in (1, 17, 1000, 5600):
        check_conv(T)
