"""Prefill GEMM throughput: bf16 torch.matmul vs fp8 torch._scaled_mm, for the model's weight shapes."""
import time
import torch

SHAPES = [(34816, 5120), (5120, 17408), (16480, 5120), (5120, 6144), (14336, 5120)]


def bench(f, n=20):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def main():
    p = torch.cuda.get_device_properties(0)
    print(p.name, p.multi_processor_count, p.total_memory // 2**30)
    for M in (512, 2048, 4096, 8192):
        tot_bf, tot_f8 = 0, 0
        for N, K in SHAPES:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            tb = bench(lambda: torch.matmul(x, w.t()))
            fl = 2 * M * N * K
            s = f"M={M} {N}x{K}: bf16 {tb*1e3:.3f} ms {fl/tb/1e12:.0f} TF"
            try:
                xf = x.to(torch.float8_e4m3fn)
                wf = w.to(torch.float8_e4m3fn)
                one = torch.ones((), device="cuda")
                sa = torch.ones(M, 1, device="cuda")
                sb = torch.ones(1, N, device="cuda")
                tf = bench(lambda: torch._scaled_mm(xf, wf.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
                s += f" | fp8 rowwise {tf*1e3:.3f} ms {fl/tf/1e12:.0f} TF"
                tot_f8 += tf
            except Exception as e:
                s += f" | fp8 failed: {str(e)[:120]}"
            tot_bf += tb
            print(s, flush=True)
        print(f"M={M}: per-layer-ish total bf16 {tot_bf*1e3:.2f} ms fp8 {tot_f8*1e3:.2f} ms")


if __name__ == "__main__":
    main()
