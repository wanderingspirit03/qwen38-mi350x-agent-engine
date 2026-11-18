import os, sys, torch, time
FP8 = torch.float8_e4m3fn
def t(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e6
for N, K in ((34816, 5120), (5120, 17408), (16480, 5120), (14336, 5120), (5120, 6144)):
    w = (torch.randn(N, K, device="cuda") * 0.5).to(FP8); ws = torch.rand(1, N, device="cuda")
    for M in (256, 384, 512, 768, 1024, 1536, 2048):
        x = (torch.randn(M, K, device="cuda")).to(FP8); xs = torch.rand(M, 1, device="cuda")
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        us = t(lambda: torch._scaled_mm(x, w.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16, out=out))
        print(f"{os.environ.get('TAG','')} N={N} K={K} M={M}: {us:7.1f} us  {N*K/us/1e6:5.2f} TB/s  {2*M*N*K/us/1e9:6.0f} TF/s", flush=True)
