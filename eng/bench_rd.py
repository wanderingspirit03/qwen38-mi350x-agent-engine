"""Plain read bandwidth on real (random) vs zero vs uninitialized buffers (w4 rd_bench): python3 -m eng.bench_rd"""
import torch

from eng.kernels import w4


def gtime(fn, reps=20):
    g = torch.cuda.CUDAGraph()
    fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = 1e9
    for _ in range(5):
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1e3 / reps)
    return best


lib = w4._build()
out = torch.zeros(4, dtype=torch.int32, device="cuda")
st = lambda: torch.cuda.current_stream().cuda_stream
for mb in (16, 47, 93, 311):
    nb = mb * 1000 * 1000 // 4096 * 4096
    for kind in ("empty", "zeros", "random"):
        bufs = []
        for _ in range(max(2, 1200 // mb)):
            b = torch.empty(nb, dtype=torch.uint8, device="cuda")
            if kind == "zeros":
                b.zero_()
            elif kind == "random":
                b.view(torch.int32).random_()
            bufs.append(b)
        res = []
        for blocks, u in ((1024, 4), (2048, 4), (4096, 1), (8192, 1)):
            def fn():
                for b in bufs:
                    lib.rd_bench(b.data_ptr(), nb, out.data_ptr(), blocks, u, st())
            t = gtime(fn, 3) / len(bufs)
            res.append(f"{blocks}x{u}: {t:6.1f} us {nb / t / 1e6:.2f} TB/s")
        print(f"{mb:4d} MB {kind:6s} " + " | ".join(res), flush=True)
        del bufs
        torch.cuda.empty_cache()
