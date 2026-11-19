"""Loads-only model of a W4A8 GEMM with independent streaming waves: wave = NT row tiles x one of KSW slices of K,
ring of D k-blocks (NT x 1 KB weights + 2 x 16 B x + 8 B scales per lane) in flight. Timing only."""
import ctypes
import subprocess
import sys

import torch

from .test_w4 import graph_time

SRC = r"""
#include <hip/hip_runtime.h>
typedef unsigned u4 __attribute__((ext_vector_type(4)));
template <int NT> struct F { u4 w[NT]; uint4 xl, xh; uint2 s; };
template <int NT, int D, int XL>
__global__ void st(const uint4* __restrict__ qw, const uint4* __restrict__ x, const uint2* __restrict__ sc, int NG,
                   int ntg, int ksw, int K, unsigned* out) {
  const int lane = threadIdx.x & 63, lg = lane >> 4, m = lane & 15;
  const int wid = blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
  int tg = wid / ksw, kp = wid % ksw;
  if (gridDim.y > 1) { tg = wid; kp = blockIdx.y; ksw = gridDim.y; }
  if (tg >= ntg) return;
  const int kb0 = kp * NG / ksw, kend = (kp + 1) * NG / ksw;
  const u4* qb = (const u4*)qw + (size_t)tg * NT * NG * 64 + lane;
  auto load = [&](F<NT>& f, int kb) {
    kb = min(kb, kend - 1);
#pragma unroll
    for (int t = 0; t < NT; t++) f.w[t] = __builtin_nontemporal_load(qb + ((size_t)t * NG + kb) * 64);
    if (XL) {
      f.xl = x[(m * K + kb * 128) / 16 + lg];
      f.xh = x[(m * K + kb * 128 + 64) / 16 + lg];
      f.s = sc[((size_t)tg * NT * NG + kb) * 4 + lg];
    }
  };
  F<NT> f[D];
#pragma unroll
  for (int i = 0; i < D; i++) load(f[i], kb0 + i);
  unsigned acc = 0;
  for (int kb = kb0; kb < kend; kb += D) {
#pragma unroll
    for (int i = 0; i < D; i++) {
      if (kb + i < kend) {
#pragma unroll
        for (int t = 0; t < NT; t++) acc ^= f[i].w[t].x ^ f[i].w[t].w;
        if (XL) acc += f[i].xl.y ^ f[i].xh.z ^ f[i].s.x;
      }
      load(f[i], kb + i + D);
    }
  }
  if (acc == 0x12345678u) out[0] = acc;
}
#define L(NT, D, XL) if (nt == NT && d == D && xl == XL) { hipLaunchKernelGGL((st<NT, D, XL>), dim3(grid), dim3(block), 0, s, (const uint4*)qw, (const uint4*)x, (const uint2*)sc, NG, ntg, ksw, K, (unsigned*)out); return 0; }
#define LD(NT, XL) L(NT, 2, XL) L(NT, 4, XL) L(NT, 8, XL)
extern "C" int launch(int nt, int d, int xl, int grid, int block, const void* qw, const void* x, const void* sc, int NG,
                      int ntg, int ksw, int K, void* out, hipStream_t s) {
  LD(1, 0) LD(2, 0) LD(4, 0) LD(1, 1) LD(2, 1) LD(4, 1)
  return 1;
}
"""


def main():
    open("/tmp/stream.hip", "w").write(SRC)
    subprocess.run(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx950", "-shared", "-fPIC", "-o", "/tmp/stream.so",
                    "/tmp/stream.hip"], check=True)
    lib = ctypes.CDLL("/tmp/stream.so")
    I, P = ctypes.c_int, ctypes.c_void_p
    lib.launch.argtypes = [I, I, I, I, I, P, P, P, I, I, I, I, P, P]
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    shapes = {"gate_up": (34816, 5120), "down": (5120, 17408), "out": (5120, 6144), "gdn_in": (16480, 5120)}
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(shapes)
    x = torch.zeros(16 * 17408 // 16 + 64, 4, dtype=torch.int32, device="cuda")
    for name in names:
        N, K = shapes[name]
        NG = K // 128
        nb = N * K // 2
        bufs = [torch.empty(nb, dtype=torch.uint8, device="cuda") for _ in range(max(3, 1500 * 2**20 // nb))]
        sc = torch.zeros(N // 16 * NG * 4 + 64, 2, dtype=torch.int32, device="cuda")
        res = []
        for xl in (0, 1):
            for nt in (1, 2, 4):
                for d in (2, 4, 8):
                    for ksw in (1, 2, 4):
                        for block in (64, 256, 512):
                            ntg = N // 16 // nt
                            nw = ntg * ksw
                            grid = -(-nw // (block // 64))
                            f = [lambda b=b: lib.launch(nt, d, xl, grid, block, b.data_ptr(), x.data_ptr(), sc.data_ptr(),
                                                        NG, ntg, ksw, K, out.data_ptr(),
                                                        torch.cuda.current_stream().cuda_stream) for b in bufs]
                            t = graph_time(f)
                            res.append((t, xl, nt, d, ksw, block))
        for v in sorted(res):
            if v[1] == 1:
                print(f"  {name} x1 nt{v[2]} d{v[3]} ksw{v[4]} blk{v[5]}: {v[0] * 1e6:.1f} us", flush=True)
        for xl in (0, 1):
            r = sorted(v for v in res if v[1] == xl)[:6]
            print(f"{name} {N}x{K} x loads {xl}: " + " | ".join(
                f"nt{v[2]} d{v[3]} ksw{v[4]} blk{v[5]}: {v[0] * 1e6:.1f} us {nb / v[0] / 1e12:.2f} TB/s" for v in r), flush=True)
        del bufs


if __name__ == "__main__":
    main()
