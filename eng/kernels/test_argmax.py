import torch
from eng import hip as H
for R, V in ((9, 248320), (1, 32768), (3, 1000)):
    x = torch.randn(R, V, device="cuda").to(torch.bfloat16)
    x[0, 5] = 50; x[0, 77] = 50
    o = torch.empty(R, dtype=torch.long, device="cuda")
    H.argmax_rows(x, o)
    ref = torch.argmax(x.float(), -1)
    mp_ = torch.arange(V, device="cuda") * 3
    o2 = torch.empty_like(o); H.argmax_rows(x, o2, mp_)
    print(R, V, torch.equal(o, ref), torch.equal(o2, ref * 3), o[0].item())
