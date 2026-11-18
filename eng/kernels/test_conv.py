import torch
from eng import hip as H
T = 1000
proj = torch.randn(T, 16480, device="cuda").to(torch.bfloat16)
hist = torch.randn(3, 10240, device="cuda").to(torch.bfloat16)
cw = (torch.randn(10240, 4, device="cuda") * 0.3).to(torch.bfloat16)
aneg = -torch.rand(48, device="cuda"); dtb = torch.randn(48, device="cuda")
y = torch.empty(T, 10240, dtype=torch.bfloat16, device="cuda"); gb = torch.empty(T, 2, 48, device="cuda")
H.gdn_conv(proj, hist, cw, aneg, dtb, y, gb)
xx = torch.cat([hist, proj[:, :10240]]).float()
acc = sum(xx[j:j + T] * cw[:, j].float() for j in range(4))
ref = (acc * torch.sigmoid(acc))
print("conv err", (y.float() - ref).abs().max().item(), "rel", ((y.float() - ref).abs().max() / ref.abs().max()).item())
a = proj[:, 10240 + 6144 + 48:].float(); b = proj[:, 10240 + 6144:10240 + 6144 + 48].float()
print("g err", (gb[:, 0] - aneg * torch.nn.functional.softplus(a + dtb)).abs().max().item(), "beta err", (gb[:, 1] - torch.sigmoid(b)).abs().max().item())
