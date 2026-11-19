"""eng.hip.add_rmsnorm and mtp_in against torch; time per call (graph of 64 calls)."""
import torch

from .. import hip as H
from .test_w4 import graph_time


def rms(x, w):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + w.float())).bfloat16()


def main():
    torch.manual_seed(0)
    Hd = 5120
    emb = torch.randn(150000, Hd, device="cuda").bfloat16()
    we, wh, w = (torch.randn(Hd, device="cuda") * 0.1).bfloat16(), (torch.randn(Hd, device="cuda") * 0.1).bfloat16(), \
        (torch.randn(Hd, device="cuda") * 0.1).bfloat16()
    for R in (1, 10):
        x = torch.randn(R, Hd, device="cuda").bfloat16()
        d = torch.randn(R, Hd, device="cuda").bfloat16()
        out = torch.empty(R, Hd, device="cuda").bfloat16()
        x2 = x.clone()
        H.add_rmsnorm(x2, d, w, out)
        xs = (x.float() + d.float()).bfloat16()
        e1 = (x2.float() - xs.float()).abs().max().item()
        e2 = (out.float() - rms(xs, w).float()).abs().max().item()
        H.add_rmsnorm(x, None, w, out)
        e3 = (out.float() - rms(x, w).float()).abs().max().item()
        tok = torch.randint(0, 150000, (R,), device="cuda")
        hid = torch.randn(R, 2 * Hd, device="cuda").bfloat16()[:, :Hd]
        mx = torch.empty(R, 2 * Hd, device="cuda").bfloat16()
        H.mtp_in(tok, emb, hid, we, wh, mx)
        e4 = (mx.float() - torch.cat([rms(emb[tok], we), rms(hid, wh)], 1).float()).abs().max().item()
        print(f"R={R}: max abs diffs add {e1}, add+norm {e2:.3g}, norm {e3:.3g}, mtp_in {e4:.3g}", flush=True)
        t1 = graph_time([lambda: [H.add_rmsnorm(x, None, w, out) for _ in range(64)]]) / 64
        t2 = graph_time([lambda: [H.add_rmsnorm(x2, d, w, out) for _ in range(64)]]) / 64
        t3 = graph_time([lambda: [H.mtp_in(tok, emb, hid, we, wh, mx) for _ in range(64)]]) / 64
        print(f"R={R}: rmsnorm {t1 * 1e6:.1f} us, add_rmsnorm {t2 * 1e6:.1f} us, mtp_in {t3 * 1e6:.1f} us", flush=True)


if __name__ == "__main__":
    main()
