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

// Grid 48 * WPH (WPH = 8 / NW workgroups per head), NW waves of 16 value columns each: wave wv of workgroup wg
// owns the value columns v0 = 16 (wg NW + wv) .. v0 + 15 of head h.
// Inputs per chunk c (64 positions) and value head h (key head h / 3):
//   w (Tp, 48, 128) bf16 [t, k]; kdT (NC, 48, 128 k, 64 t) bf16; uT (NC, 48, 128 v, 64 t) bf16;
//   qn (Tp, 16, 128) bf16 (normalized, scaled q); M (NC, 48, 64 t, 64 t') bf16 = q_t.k_t' exp(G_t - G_t'), t' <= t,
//   zero above the diagonal; G (48, Tp) fp32, the cumulative log decay inside each chunk.
// state: S[h, k, v] at state + h s_h + k s_k + v s_v, fp32, updated in place. NC must be even.
// Per chunk: o = exp(G) (q S) + M vn, vn = u - w S, S = exp(G_last) S + kd^T vn. o is computed transposed
// (o^T = S^T q^T + vn^T M^T: the B-operand registers of S and vn are the A operands), so each lane holds 4
// consecutive v of one t; out row t (bf16) at out + t * so + h * 128.
// The chunk's w, kd, q and M (56 KB) are read from global memory once per workgroup, staged in registers one
// chunk ahead, and shared through two LDS buffers (XOR-swizzled 16-byte units: conflict-free fragment reads).
// The vector memory counter is in order, so the u loads are issued before the chunk loads of a step.
template <int NW>
__global__ __launch_bounds__(64 * NW) void gdn_scan_kernel(const bf16* __restrict__ w, const bf16* __restrict__ uT,
                                                       const bf16* __restrict__ kdT, const bf16* __restrict__ qn,
                                                       const bf16* __restrict__ Mm, const float* __restrict__ G,
                                                       float* __restrict__ state, long s_h, long s_k, long s_v,
                                                       bf16* __restrict__ out, long so, int T, int NC, int Tp) {
  __shared__ __attribute__((aligned(16))) bf16 lw[2][64 * 128];
  __shared__ __attribute__((aligned(16))) bf16 lk[2][128 * 64];
  __shared__ __attribute__((aligned(16))) bf16 lq[2][64 * 128];
  __shared__ __attribute__((aligned(16))) bf16 lm[2][64 * 64];
  constexpr int NTH = 64 * NW, WPH = 8 / NW, NU = 1024 / NTH, NS = 3 * NU + NU / 2;   // NS 16-byte units per thread
  // Workgroups are dispatched round-robin over the 8 XCDs (b & 7): the WPH workgroups of a head are put on one XCD
  // so that their shared chunk reads hit its L2.
  const int b0 = blockIdx.x, sl = b0 >> 3;
  const int h = (sl / WPH) * 8 + (b0 & 7), hk = h / 3, tid = threadIdx.x;
  const int lane = tid & 63, r = lane & 15, q = lane >> 4, wv = tid >> 6;
  const int v0 = 16 * ((sl % WPH) * NW + wv);
  f32x4 S[8];   // S[mt][j] = S[k = 16 mt + 4 q + j, v = v0 + r]
#pragma unroll
  for (int mt = 0; mt < 8; mt++)
#pragma unroll
    for (int j = 0; j < 4; j++) S[mt][j] = state[h * s_h + (v0 + r) * s_v + (16 * mt + 4 * q + j) * s_k];

  // 3584 units per chunk: 1024 of w, 1024 of kd, 1024 of q, 512 of M.
  auto gload = [&](u32x4 (&stg)[NS], int c) {
#pragma unroll
    for (int i = 0; i < NS; i++) {
      int u = tid + NTH * (i % NU);
      const bf16* p;
      if (i < NU) p = w + ((size_t)(c * SC + (u >> 4)) * SHV + h) * SD + (u & 15) * 8;
      else if (i < 2 * NU) p = kdT + (((size_t)c * SHV + h) * SD + (u >> 3)) * SC + (u & 7) * 8;
      else if (i < 3 * NU) p = qn + ((size_t)(c * SC + (u >> 4)) * 16 + hk) * SD + (u & 15) * 8;
      else p = Mm + (((size_t)c * SHV + h) * SC + (u >> 3)) * SC + (u & 7) * 8;
      stg[i] = *(const u32x4*)p;
    }
  };
  auto lstore = [&](const u32x4 (&stg)[NS], int b) {
#pragma unroll
    for (int i = 0; i < NS; i++) {
      int u = tid + NTH * (i % NU);
      int r16 = u >> 4, r8 = u >> 3;
      bf16* p;
      if (i < NU) p = &lw[b][r16 * 128 + (((u & 15) ^ (r16 & 15)) * 8)];
      else if (i < 2 * NU) p = &lk[b][r8 * 64 + (((u & 7) ^ ((r8 >> 1) & 7)) * 8)];
      else if (i < 3 * NU) p = &lq[b][r16 * 128 + (((u & 15) ^ (r16 & 15)) * 8)];
      else p = &lm[b][r8 * 64 + (((u & 7) ^ ((r8 >> 1) & 7)) * 8)];
      *(u32x4*)p = stg[i];
    }
  };
  struct Step {
    u32x2 u[4];   // u[t = 16 mt + 4 q + j, v0 + r]
    float eg[4];  // G[t = 16 mt + r]
    float gl;     // G at the chunk end
  };
  auto uload = [&](Step& st, int c) {
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
      st.u[mt] = *(const u32x2*)(uT + (((size_t)c * SHV + h) * SD + v0 + r) * SC + 16 * mt + 4 * q);
      st.eg[mt] = G[(size_t)h * Tp + c * SC + 16 * mt + r];
    }
    st.gl = G[(size_t)h * Tp + c * SC + SC - 1];
  };

  // One step. Each phase loads all its LDS fragments first (sched_barrier keeps the compiler from moving the reads
  // next to their MFMAs, which would expose the LDS latency once per MFMA).
  auto compute = [&](const Step& st, int c, int b) {
    float e = __expf(st.gl);
    bf16x8 fr[16];
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
      for (int kb = 0; kb < 4; kb++) fr[4 * mt + kb] = *(const bf16x8*)(&lw[b][(16 * mt + r) * 128 + (((4 * kb + q) ^ r) * 8)]);
    bf16x8 Sb[4];
    {
      u32x2 Sp[8];
#pragma unroll
      for (int mt = 0; mt < 8; mt++) Sp[mt] = pack4(S[mt]);
#pragma unroll
      for (int kb = 0; kb < 4; kb++) Sb[kb] = c2b(Sp[2 * kb], Sp[2 * kb + 1]);
    }
    __builtin_amdgcn_sched_barrier(0);
    f32x4 vn[4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) vn[mt] = (f32x4){0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kb = 0; kb < 4; kb++)
#pragma unroll
      for (int mt = 0; mt < 4; mt++) vn[mt] = MFMA(fr[4 * mt + kb], Sb[kb], vn[mt]);
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int mt = 0; mt < 8; mt++)
#pragma unroll
      for (int i = 0; i < 2; i++) fr[2 * mt + i] = *(const bf16x8*)(&lk[b][(16 * mt + r) * 64 + (((4 * i + q) ^ (r >> 1)) * 8)]);
#pragma unroll
    for (int mt = 0; mt < 8; mt++) S[mt] *= e;
    bf16x8 vb[2];
    {
      u32x2 vp[4];
#pragma unroll
      for (int mt = 0; mt < 4; mt++) vp[mt] = pack4(unpack4(st.u[mt]) - vn[mt]);
#pragma unroll
      for (int i = 0; i < 2; i++) vb[i] = c2b(vp[2 * i], vp[2 * i + 1]);
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
      for (int mt = 0; mt < 8; mt++) S[mt] = MFMA(fr[2 * mt + i], vb[i], S[mt]);
    __builtin_amdgcn_sched_barrier(0);
    // o^T tiles (v0 .. v0 + 15) x (t = 16 mt ..): exp(G_t) S^T q^T + vn^T M^T.
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
      for (int kb = 0; kb < 4; kb++) fr[4 * mt + kb] = *(const bf16x8*)(&lq[b][(16 * mt + r) * 128 + (((4 * kb + q) ^ r) * 8)]);
    bf16x8 fm[6];
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
      for (int i = 0; i < 2; i++)
        if (i == 0 || mt >= 2)   // M is zero for t' > t
          fm[mt < 2 ? mt : 2 * mt - 2 + i] = *(const bf16x8*)(&lm[b][(16 * mt + r) * 64 + (((4 * i + q) ^ (r >> 1)) * 8)]);
    __builtin_amdgcn_sched_barrier(0);
    f32x4 o[4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) o[mt] = (f32x4){0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kb = 0; kb < 4; kb++)
#pragma unroll
      for (int mt = 0; mt < 4; mt++) o[mt] = MFMA(Sb[kb], fr[4 * mt + kb], o[mt]);
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
      o[mt] *= __expf(st.eg[mt]);
      o[mt] = MFMA(vb[0], fm[mt < 2 ? mt : 2 * mt - 2], o[mt]);
      if (mt >= 2) o[mt] = MFMA(vb[1], fm[2 * mt - 1], o[mt]);
      int t = c * SC + 16 * mt + r;
      if (t < T) *(u32x2*)(out + (size_t)t * so + h * SD + v0 + 4 * q) = pack4(o[mt]);
    }
  };

  // Step c: barrier, compute c from LDS buffer c & 1, then u of chunk c + 1, chunk c + 1 (staged in registers
  // during step c - 1) into the other LDS buffer, and the chunk loads of c + 2.
  u32x4 stg[NS];
  Step ua, ub;
  gload(stg, 0);
  lstore(stg, 0);
  gload(stg, min(1, NC - 1));
  uload(ua, 0);
  for (int c = 0; c < NC; c += 2) {
    __syncthreads();
    compute(ua, c, 0);
    uload(ub, c + 1);
    lstore(stg, 1);
    gload(stg, min(c + 2, NC - 1));
    __syncthreads();
    compute(ub, c + 1, 1);
    uload(ua, min(c + 2, NC - 1));
    lstore(stg, 0);
    gload(stg, min(c + 3, NC - 1));
  }
#pragma unroll
  for (int mt = 0; mt < 8; mt++)
#pragma unroll
    for (int j = 0; j < 4; j++) state[h * s_h + (v0 + r) * s_v + (16 * mt + 4 * q + j) * s_k] = S[mt][j];
}

// nw: waves per workgroup (8, 4 or 2; 8 / nw workgroups per head).
extern "C" int gdn_scan(int nw, const void* w, const void* uT, const void* kdT, const void* qn, const void* M,
                        const void* G, void* state, long s_h, long s_k, long s_v, void* out, long so, int T, int NC,
                        int Tp, hipStream_t s) {
  auto run = [&](auto kern) {
    hipLaunchKernelGGL(kern, dim3(SHV * 8 / nw), dim3(64 * nw), 0, s, (const bf16*)w, (const bf16*)uT,
                       (const bf16*)kdT, (const bf16*)qn, (const bf16*)M, (const float*)G, (float*)state, s_h, s_k,
                       s_v, (bf16*)out, so, T, NC, Tp);
  };
  if (nw == 8) run(gdn_scan_kernel<8>);
  else if (nw == 4) run(gdn_scan_kernel<4>);
  else run(gdn_scan_kernel<2>);
  return (int)hipGetLastError();
}
