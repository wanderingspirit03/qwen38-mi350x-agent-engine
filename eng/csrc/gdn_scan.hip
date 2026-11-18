// Sequential part of the chunked gated delta rule prefill (eng/kernels/gdn_prefill.py): the state over chunks of 64.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __bf16 bf16;
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));

#define SHV 48
#define SD 128
#define SC 64

static __device__ __forceinline__ u32x2 pack4(f32x4 x) {
  return __builtin_bit_cast(u32x2, __builtin_convertvector(x, bf16x4));
}
static __device__ __forceinline__ f32x4 unpack4(u32x2 x) {
  return (f32x4){__uint_as_float(x[0] << 16), __uint_as_float(x[0] & 0xffff0000u), __uint_as_float(x[1] << 16),
                 __uint_as_float(x[1] & 0xffff0000u)};
}
// Two 16x16 accumulator tiles (rows 0-15 in a, 16-31 in b; lane (r, q) holds rows 4q..4q+3 of column r), as bf16,
// -> the B operand of v_mfma_f32_16x16x32_bf16 for those 32 rows (lane (r, q) holds rows 8q..8q+7 of column r).
static __device__ __forceinline__ bf16x8 c2b(u32x2 a, u32x2 b) {
#pragma unroll
  for (int i = 0; i < 2; i++) {
    auto s = __builtin_amdgcn_permlane32_swap(a[i], b[i], false, false);
    a[i] = s[0]; b[i] = s[1];
  }
#pragma unroll
  for (int i = 0; i < 2; i++) {
    auto s = __builtin_amdgcn_permlane16_swap(a[i], b[i], false, false);
    a[i] = s[0]; b[i] = s[1];
  }
  return __builtin_bit_cast(bf16x8, (u32x4){a[0], a[1], b[0], b[1]});
}
#define MFMA(a, b, c) __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0)

// Grid 48 (one workgroup per head), block 256 = 4 waves of 32 value columns (v0 = 32 * wave).
// Inputs: w (Tp, 48, 128) bf16 [t, k]; kdT (NC, 48, 128 k, 64 t) bf16; uT (NC, 48, 128 v, 64 t) bf16 and
// G (48, Tp) fp32 (cumulative log decay inside each chunk; the chunk's last entry is used).
// state: S[h, k, v] at state + h s_h + k s_k + v s_v, fp32, updated in place. NC must be even.
// Outputs: hT (NC, 48, 128 v, 128 k) bf16, S at the chunk start; vnT (NC, 48, 128 v, 64 t) bf16. The chunk's w and kd (32 KB) are read from global memory once per head and shared through
// LDS (two buffers, XOR-swizzled 16-byte units: conflict-free fragment reads); the global loads run two chunks
// ahead in registers.
template <int DBG, int NT, int NW>
__global__ __launch_bounds__(64 * NW) void gdn_scan_kernel(const bf16* __restrict__ w, const bf16* __restrict__ uT,
                                                       const bf16* __restrict__ kdT, const float* __restrict__ G,
                                                       float* __restrict__ state, long s_h, long s_k, long s_v,
                                                       bf16* __restrict__ hT, bf16* __restrict__ vnT, int NC, int Tp,
                                                       long long* __restrict__ tdbg) {
  __shared__ __attribute__((aligned(16))) bf16 lw[2][64 * 128];
  __shared__ __attribute__((aligned(16))) bf16 lk[2][128 * 64];
  constexpr int NTH = 64 * NW, WPH = 8 / (NT * NW);   // WPH workgroups per head
  int h = blockIdx.x / WPH, tid = threadIdx.x;
  int lane = tid & 63, r = lane & 15, q = lane >> 4, wv = tid >> 6;
  int v0 = ((blockIdx.x % WPH) * NW + wv) * 16 * NT;
  f32x4 S[NT][8];   // S[n][mt][j] = S[k = 16 mt + 4 q + j, v = v0 + 16 n + r]
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int mt = 0; mt < 8; mt++)
#pragma unroll
      for (int j = 0; j < 4; j++) S[n][mt][j] = state[h * s_h + (v0 + 16 * n + r) * s_v + (16 * mt + 4 * q + j) * s_k];

  constexpr int NL = 1024 / NTH;
  u32x4 gw0[NL], gk0[NL], gw1[NL], gk1[NL];
  auto gload = [&](u32x4 (&dw)[NL], u32x4 (&dk)[NL], int c) {
#pragma unroll
    for (int i = 0; i < NL; i++) {
      int u = tid + NTH * i;
      dw[i] = *(const u32x4*)(w + ((size_t)(c * SC + (u >> 4)) * SHV + h) * SD + (u & 15) * 8);
      dk[i] = *(const u32x4*)(kdT + (((size_t)c * SHV + h) * SD + (u >> 3)) * SC + (u & 7) * 8);
    }
  };
  auto lstore = [&](const u32x4 (&dw)[NL], const u32x4 (&dk)[NL], int b) {
#pragma unroll
    for (int i = 0; i < NL; i++) {
      int u = tid + NTH * i;
      int rw = u >> 4, rk = u >> 3;
      *(u32x4*)(&lw[b][rw * 128 + (((u & 15) ^ (rw & 15)) * 8)]) = dw[i];
      *(u32x4*)(&lk[b][rk * 64 + (((u & 7) ^ ((rk >> 1) & 7)) * 8)]) = dk[i];
    }
  };
  u32x2 ua[NT][4], ub[NT][4];
  float gla, glb;
  auto uload = [&](u32x2 (&u)[NT][4], float& gl, int c) {
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int mt = 0; mt < 4; mt++)
        u[n][mt] = *(const u32x2*)(uT + (((size_t)c * SHV + h) * SD + v0 + 16 * n + r) * SC + 16 * mt + 4 * q);
    gl = G[(size_t)h * Tp + c * SC + SC - 1];
  };
  auto compute = [&](const u32x2 (&u)[NT][4], float gl, int c, int b) {
    // All LDS fragments first, so the reads overlap each other and the conversions of S.
    bf16x8 aw[4][4], ak[8][2];
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
      for (int kb = 0; kb < 4; kb++) aw[mt][kb] = *(const bf16x8*)(&lw[b][(16 * mt + r) * 128 + (((4 * kb + q) ^ r) * 8)]);
#pragma unroll
    for (int mt = 0; mt < 8; mt++)
#pragma unroll
      for (int i = 0; i < 2; i++) ak[mt][i] = *(const bf16x8*)(&lk[b][(16 * mt + r) * 64 + (((4 * i + q) ^ (r >> 1)) * 8)]);
    float e = __expf(gl);
    bf16x8 Sb[NT][4];
#pragma unroll
    for (int n = 0; n < NT; n++) {
      u32x2 Sp[8];
#pragma unroll
      for (int mt = 0; mt < 8; mt++) Sp[mt] = pack4(S[n][mt]);
#pragma unroll
      for (int kb = 0; kb < 4; kb++) Sb[n][kb] = c2b(Sp[2 * kb], Sp[2 * kb + 1]);
    }
    f32x4 vn[NT][4];
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int mt = 0; mt < 4; mt++) vn[n][mt] = (f32x4){0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kb = 0; kb < 4; kb++)
#pragma unroll
      for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int n = 0; n < NT; n++) vn[n][mt] = MFMA(aw[mt][kb], Sb[n][kb], vn[n][mt]);
    // S at the chunk start goes out while the MFMAs run.
#pragma unroll
    for (int n = 0; n < NT; n++) {
      bf16* hc = hT + (((size_t)c * SHV + h) * SD + v0 + 16 * n + r) * SD + 8 * q;
#pragma unroll
      for (int kb = 0; kb < 4; kb++)
        if (!(DBG & 1)) __builtin_nontemporal_store(Sb[n][kb], (bf16x8*)(hc + 32 * kb));
#pragma unroll
      for (int mt = 0; mt < 8; mt++) S[n][mt] *= e;
    }
    // Keeps the compiler from hoisting the use of u (and so its s_waitcnt, which also waits for the stores
    // issued before the u loads) to the top of the step.
    __builtin_amdgcn_sched_barrier(0);
    bf16x8 vb[NT][2];
#pragma unroll
    for (int n = 0; n < NT; n++) {
      u32x2 vp[4];
#pragma unroll
      for (int mt = 0; mt < 4; mt++) vp[mt] = pack4(unpack4(u[n][mt]) - vn[n][mt]);
#pragma unroll
      for (int i = 0; i < 2; i++) vb[n][i] = c2b(vp[2 * i], vp[2 * i + 1]);
    }
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
      for (int mt = 0; mt < 8; mt++)
#pragma unroll
        for (int n = 0; n < NT; n++) S[n][mt] = MFMA(ak[mt][i], vb[n][i], S[n][mt]);
#pragma unroll
    for (int n = 0; n < NT; n++) {
      bf16* vc = vnT + (((size_t)c * SHV + h) * SD + v0 + 16 * n + r) * SC + 8 * q;
#pragma unroll
      for (int i = 0; i < 2; i++)
        if (!(DBG & 1)) __builtin_nontemporal_store(vb[n][i], (bf16x8*)(vc + 32 * i));
    }
  };
  auto stamp = [&](int c, int k) {
    if ((DBG & 2) && tid == 0 && c < 64) {
      long long t;
      asm volatile("s_memtime %0\n\ts_waitcnt lgkmcnt(0)" : "=s"(t));
      tdbg[(blockIdx.x * 64 + c) * 4 + k] = t;
    }
  };

  gload(gw0, gk0, 0);
  gload(gw1, gk1, min(1, NC - 1));
  uload(ua, gla, 0);
  uload(ub, glb, min(1, NC - 1));
  lstore(gw0, gk0, 0);
  for (int c = 0; c < NC; c += 2) {
    stamp(c, 0);
    gload(gw0, gk0, min(c + 2, NC - 1));
    __syncthreads();
    stamp(c, 1);
    compute(ua, gla, c, 0);
    stamp(c, 2);
    uload(ua, gla, min(c + 2, NC - 1));
    lstore(gw1, gk1, 1);
    stamp(c, 3);
    gload(gw1, gk1, min(c + 3, NC - 1));
    __syncthreads();
    compute(ub, glb, c + 1, 1);
    uload(ub, glb, min(c + 3, NC - 1));
    lstore(gw0, gk0, 0);
  }
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int mt = 0; mt < 8; mt++)
#pragma unroll
      for (int j = 0; j < 4; j++) state[h * s_h + (v0 + 16 * n + r) * s_v + (16 * mt + 4 * q + j) * s_k] = S[n][mt][j];
}

extern "C" int gdn_scan(int dbg, void* tdbg, const void* w, const void* uT, const void* kdT, const void* G,
                        void* state, long s_h, long s_k, long s_v, void* hT, void* vnT, int NC, int Tp,
                        hipStream_t s) {
  auto run = [&](auto kern, int nt, int nw) {
    hipLaunchKernelGGL(kern, dim3(SHV * 8 / (nt * nw)), dim3(64 * nw), 0, s, (const bf16*)w, (const bf16*)uT,
                       (const bf16*)kdT, (const float*)G, (float*)state, s_h, s_k, s_v, (bf16*)hT, (bf16*)vnT, NC,
                       Tp, (long long*)tdbg);
  };
  switch (dbg) {
    case 1: run(gdn_scan_kernel<1, 1, 4>, 1, 4); break;
    case 2: run(gdn_scan_kernel<2, 1, 4>, 1, 4); break;
    default: run(gdn_scan_kernel<0, 1, 4>, 1, 4);
  }
  return (int)hipGetLastError();
}
