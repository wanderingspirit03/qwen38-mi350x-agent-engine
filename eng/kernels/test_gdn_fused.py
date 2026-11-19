"""gdn_fused (one kernel per GDN layer) vs gdn_prep + gdn_verify_q: outputs and time per layer (graphs of 48 layers).
python3 -m eng.kernels.test_gdn_fused"""
import torch

from eng import hip as H
from eng.bench_round import graph, tm
from eng.kernels import w4a8

torch.manual_seed(0)
dev = "cuda"
NH, DK, NCH, PROJ = 48, 128, 10240, 16480


def layer():
    return dict(hist=torch.randn(3, NCH, device=dev).bfloat16(), cw=(torch.randn(NCH, 4, device=dev) * 0.5).bfloat16(),
                aneg=-torch.rand(NH, device=dev) * 2, dtb=torch.randn(NH, device=dev),
                state=torch.randn(NH, DK, DK, device=dev) * 0.3, gw=(1 + 0.2 * torch.randn(DK, device=dev)).bfloat16())


def bufs(T):
    return dict(qk=torch.zeros(T, 32, DK, device=dev), v=torch.zeros(T, NH, DK, device=dev),
                gb=torch.zeros(T, 2, NH, device=dev), xa=w4a8.new_xa(T, 6144, dev))


def old(proj, Ly, b, o):
    H.gdn_prep(proj, Ly["hist"], Ly["cw"], Ly["aneg"], Ly["dtb"], b["qk"], b["v"], b["gb"])
    H.gdn_verify_q(b["qk"], b["v"], b["gb"], Ly["state"], o, proj, Ly["gw"], b["xa"])


def new(proj, Ly, b, o=None):
    H.gdn_fused(proj, Ly["hist"], Ly["cw"], Ly["aneg"], Ly["dtb"], Ly["state"], Ly["gw"], b["qk"], b["v"], b["gb"], b["xa"])


def deq(xa, T):
    return xa["xq"][:T].float().view(T, NH, DK) * xa["xg"][:T, :NH, 0:1].float() * 16


def main():
    for T in (1, 2, 5, 10, 16):
        proj = torch.randn(T, PROJ + 32, device=dev).bfloat16()[:, :PROJ]
        Ly = layer()
        a, b = bufs(T), bufs(T)
        o = torch.zeros(16, NH, DK, device=dev)
        old(proj, Ly, a, o)
        new(proj, Ly, b)
        torch.cuda.synchronize()
        e = {k: ((a[k] - b[k]).abs().max() / a[k].abs().max()).item() for k in ("qk", "v", "gb")}
        da, db = deq(a["xa"], T), deq(b["xa"], T)
        dq = (a["xa"]["xq"][:T].int() - b["xa"]["xq"][:T].int()).abs()
        zp = ((a["xa"]["xzp"][:T].float() - b["xa"]["xzp"][:T].float()).abs().max()).item()
        print(f"T={T:2d} rel err qk {e['qk']:.1e} v {e['v']:.1e} gb {e['gb']:.1e}; out rel err "
              f"{((da - db).abs().max() / da.abs().max()).item():.1e}, int8 diff max {dq.max().item()} "
              f"(share != {(dq > 0).float().mean().item():.1e}), zp diff {zp:.2e}", flush=True)
    L = 48
    Ls = [layer() for _ in range(L)]
    o = torch.zeros(16, NH, DK, device=dev)
    for T in (1, 4, 8, 10, 12, 16):
        proj = torch.randn(T, PROJ, device=dev).bfloat16()
        b = bufs(T)
        res = [f"T={T:2d}"]
        for name, f in (("prep+verify_q", old), ("fused", new)):
            g = graph(lambda: [f(proj, Ls[l], b, o) for l in range(L)])
            res.append(f"{name} {tm(g.replay, 50) * 1e3 / L:.2f}")
        print("  ".join(res) + "  us per layer", flush=True)


if __name__ == "__main__":
    main()
