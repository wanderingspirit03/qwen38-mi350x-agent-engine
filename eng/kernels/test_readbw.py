"""Peak read bandwidth of this GPU: a plain read kernel (sum of uint4 loads) over buffers that do not fit the
256 MB last-level cache, for several grid / block / unroll settings. Not part of the engine."""
import ctypes
import subprocess

import torch

from .test_w4 import graph_time

SRC = r"""
#include <hip/hip_runtime.h>
template <int U>
__global__ void rd(const uint4* __restrict__ p, size_t n, unsigned* out) {
  unsigned acc = 0;
  size_t stride = (size_t)gridDim.x * blockDim.x * U;
  for (size_t i = (size_t)blockIdx.x * blockDim.x * U + threadIdx.x; i < n; i += stride) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; u++) v[u] = i + u * blockDim.x < n ? p[i + u * blockDim.x] : uint4{0, 0, 0, 0};
#pragma unroll
    for (int u = 0; u < U; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x12345678u) out[0] = acc;
}
extern "C" int launch(int u, int grid, int block, const void* p, size_t n, void* out, hipStream_t s) {
  if (u == 1) hipLaunchKernelGGL(rd<1>, dim3(grid), dim3(block), 0, s, (const uint4*)p, n, (unsigned*)out);
  if (u == 4) hipLaunchKernelGGL(rd<4>, dim3(grid), dim3(block), 0, s, (const uint4*)p, n, (unsigned*)out);
  if (u == 8) hipLaunchKernelGGL(rd<8>, dim3(grid), dim3(block), 0, s, (const uint4*)p, n, (unsigned*)out);
  if (u == 16) hipLaunchKernelGGL(rd<16>, dim3(grid), dim3(block), 0, s, (const uint4*)p, n, (unsigned*)out);
  return 0;
}
"""


def main():
    open("/tmp/rdbw.hip", "w").write(SRC)
    subprocess.run(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx950", "-shared", "-fPIC", "-o", "/tmp/rdbw.so",
                    "/tmp/rdbw.hip"], check=True)
    lib = ctypes.CDLL("/tmp/rdbw.so")
    lib.launch.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                           ctypes.c_void_p]
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    for mb in (89, 16):
        nb = mb << 20
        bufs = [torch.empty(nb, dtype=torch.uint8, device="cuda") for _ in range(max(4, 1200 // mb))]
        best = None
        for u in (1, 4, 8, 16):
            for block in (64, 128, 256, 512, 1024):
                for grid in (256, 512, 1024, 2048):
                    f = [lambda b=b: lib.launch(u, grid, block, b.data_ptr(), nb // 16, out.data_ptr(),
                                                torch.cuda.current_stream().cuda_stream) for b in bufs]
                    t = graph_time(f)
                    print(f"  {mb} MB u={u} block={block} grid={grid}: {nb / t / 1e12:.2f} TB/s", flush=True)
                    if best is None or t < best[0]:
                        best = (t, u, block, grid)
        t, u, block, grid = best
        print(f"read {mb} MB: best {t * 1e6:.1f} us = {nb / t / 1e12:.2f} TB/s (unroll {u}, block {block}, grid {grid})",
              flush=True)
        del bufs


if __name__ == "__main__":
    main()
