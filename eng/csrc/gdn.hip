// Gated DeltaNet decode/verify kernels for T <= 16 new positions of one sequence (batch 1).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

typedef __hip_bfloat16 bf16;
static __device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }
static __device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16(x)); }

#define NCH 10240
#define QKC 2048
#define VD 6144
#define NH 48
#define DK 128

// conv history (3 rows) followed by the projected rows
static __device__ __forceinline__ float xrow(const bf16* proj, int pstride, const bf16* hist, int r, int c) {
  return r < 3 ? bf(hist[r * NCH + c]) : bf(proj[(r - 3) * pstride + c]);
}

// grid (T, 81), block 128. Groups 0..15: q heads, 16..31: k heads, 32..79: v heads, 80: g and beta.
// Writes qk (T, 32, 128) fp32 (q scaled by 1/sqrt(128), both L2-normalized), v (T, 48, 128) fp32,
// gb (T, 2, 48) fp32 (g = log decay, beta).
__global__ void gdn_prep_kernel(const bf16* __restrict__ proj, int pstride, const bf16* __restrict__ hist,
                                const bf16* __restrict__ cw, const float* __restrict__ aneg,
                                const float* __restrict__ dtb, float* __restrict__ qk, float* __restrict__ v,
                                float* __restrict__ gb) {
  int t = blockIdx.x, grp = blockIdx.y, i = threadIdx.x;
  int T = gridDim.x;
  if (grp == 80) {
    if (i < NH) {
      float b = bf(proj[t * pstride + NCH + VD + i]);
      float a = bf(proj[t * pstride + NCH + VD + NH + i]);
      float x = a + dtb[i];
      float sp = x > 20.f ? x : log1pf(expf(x));
      gb[(t * 2 + 0) * NH + i] = aneg[i] * sp;
      gb[(t * 2 + 1) * NH + i] = 1.f / (1.f + expf(-b));
    }
    return;
  }
  int c = grp * 128 + i;
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < 4; j++) acc += xrow(proj, pstride, hist, t + j, c) * bf(cw[c * 4 + j]);
  float y = rbf(acc / (1.f + expf(-acc)));
  if (grp >= 32) { v[(t * NH + (grp - 32)) * DK + i] = y; return; }
  __shared__ float red[2];
  float s = y * y;
  for (int o = 32; o > 0; o >>= 1) s += __shfl_xor(s, o);
  if ((i & 63) == 0) red[i >> 6] = s;
  __syncthreads();
  float tot = red[0] + red[1];
  float r = rsqrtf(tot + 1e-6f);
  if (grp < 16) r *= 0.08838834764831845f;
  qk[(t * 32 + grp) * DK + i] = y * r;
}

// The value column of a thread group of 4: thread part p holds S[k] for k in [32p, 32p + 32).
#define NP 8    // threads per value column (k parts)
#define KP 16   // k per thread (DK / NP)
#define VB 16   // value columns per block (block = VB * NP threads = DK: the loaders below rely on it)
static __device__ __forceinline__ float dpp_xor1(float x) {
  return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), 0xB1, 0xF, 0xF, true));
}
static __device__ __forceinline__ float dpp_xor2(float x) {
  return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), 0x4E, 0xF, 0xF, true));
}
// sum over the NP = 8 lanes of a value column
static __device__ __forceinline__ float sum_np(float x) {
  x += dpp_xor1(x);
  x += dpp_xor2(x);
  if (NP == 8) x += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(x), 0x101F));   // + lane ^ 4
  return x;
}

// S_t = d S_{t-1} + k delta^T, delta = (v - d S_{t-1}^T k) beta, o = S_t^T q = d S_{t-1}^T q + (k.q) delta:
// both dot products with S_{t-1} in one pass, then the update (one reduction per step).
static __device__ __forceinline__ float delta_step(float* S, const float* q, const float* k, float vv, float g,
                                                   float beta, float kq) {
  float decay = __expf(g);
  float a0 = 0.f, a1 = 0.f, b0 = 0.f, b1 = 0.f;
#pragma unroll
  for (int i = 0; i < KP; i += 2) {
    a0 += S[i] * k[i]; a1 += S[i + 1] * k[i + 1];
    b0 += S[i] * q[i]; b1 += S[i + 1] * q[i + 1];
  }
  float kv = sum_np(a0 + a1), sq = sum_np(b0 + b1);
  float delta = (vv - decay * kv) * beta;
#pragma unroll
  for (int i = 0; i < KP; i++) S[i] = S[i] * decay + k[i] * delta;
  return decay * sq + kq * delta;
}

static __device__ __forceinline__ void load_state(float* S, const float* sp) {
  const float4* p4 = reinterpret_cast<const float4*>(sp);
#pragma unroll
  for (int i = 0; i < KP / 4; i++) {
    float4 x = p4[i];
    S[4 * i] = x.x; S[4 * i + 1] = x.y; S[4 * i + 2] = x.z; S[4 * i + 3] = x.w;
  }
}

// Shared by verify and commit: block (h, vb) runs n steps for value columns vb*VB.. of head h.
// o_out (if not null): (T, 48, 128) fp32 raw outputs. state_out (if not null): the state after n steps.
static __device__ __forceinline__ void run_steps(int n, int h, int vb, const float* __restrict__ qk,
                                                 const float* __restrict__ v, const float* __restrict__ gb,
                                                 const float* __restrict__ state, float* __restrict__ o_out,
                                                 float* __restrict__ state_out) {
  int tid = threadIdx.x, lc = tid / NP, part = tid % NP, kh = h / 3, col = vb * VB + lc;
  // [t][part][KP + 4]: the k-parts start (KP + 4) * 4 bytes apart, on different LDS banks
  __shared__ __attribute__((aligned(16))) float sq[16][NP][KP + 4], sk[16][NP][KP + 4];
  __shared__ float sv[16][VB], sg[16], sb[16];
  float S[KP];
  load_state(S, state + ((size_t)h * DK + col) * DK + part * KP);
  // every load first and nothing under a branch, so the loads and the 16 reductions below overlap (rows >= n
  // repeat row n - 1 and are not used)
  constexpr int RV = 16 * VB / DK;   // v loads per thread (DK / VB rows per pass)
  float rq[16], rk[16], rv[RV];
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const int jj = min(j, n - 1);
    rq[j] = qk[(jj * 32 + kh) * DK + tid];
    rk[j] = qk[(jj * 32 + 16 + kh) * DK + tid];
  }
#pragma unroll
  for (int j = 0; j < RV; j++) {
    int t = min(j * (DK / VB) + tid / VB, n - 1);
    rv[j] = v[(t * NH + h) * DK + vb * VB + tid % VB];
  }
  const int tg = min(tid, n - 1);
  float rg = gb[(tg * 2) * NH + h], rb = gb[(tg * 2 + 1) * NH + h];
#pragma unroll
  for (int j = 0; j < 16; j++) { sq[j][tid / KP][tid % KP] = rq[j]; sk[j][tid / KP][tid % KP] = rk[j]; }
#pragma unroll
  for (int j = 0; j < RV; j++) sv[j * (DK / VB) + tid / VB][tid % VB] = rv[j];
  if (tid < 16) { sg[tid] = rg; sb[tid] = rb; }
  __shared__ float skq[16][2];
  float x[16];
#pragma unroll
  for (int j = 0; j < 16; j++) x[j] = rq[j] * rk[j];
#pragma unroll
  for (int m = 32; m > 0; m >>= 1)
#pragma unroll
    for (int j = 0; j < 16; j++) x[j] += __shfl_xor(x[j], m);
  if ((tid & 63) == 0)
#pragma unroll
    for (int j = 0; j < 16; j++) skq[j][tid >> 6] = x[j];
  __syncthreads();
  for (int t = 0; t < n; t++) {
    float o = delta_step(S, sq[t][part], sk[t][part], sv[t][lc], sg[t], sb[t], skq[t][0] + skq[t][1]);
    if (o_out && part == 0) o_out[((size_t)t * NH + h) * DK + col] = o;
  }
  if (state_out) {
    float4* p4 = reinterpret_cast<float4*>(state_out + ((size_t)h * DK + col) * DK + part * KP);
#pragma unroll
    for (int i = 0; i < KP / 4; i++) p4[i] = make_float4(S[4 * i], S[4 * i + 1], S[4 * i + 2], S[4 * i + 3]);
  }
}

// verify: grid (48, 128 / VB), block VB * NP. state (48, 128 v, 128 k) fp32, read only. o (T, 48, 128) fp32.
__global__ __launch_bounds__(VB * NP) void gdn_verify_kernel(int T, const float* __restrict__ qk,
                                                           const float* __restrict__ v, const float* __restrict__ gb,
                                                           const float* __restrict__ state, float* __restrict__ o) {
  run_steps(T, blockIdx.x, blockIdx.y, qk, v, gb, state, o, nullptr);
}

typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));

// One kernel per GDN layer and verify: conv + SiLU + L2 norm (gdn_prep), the recurrence over n <= 16 rows in chunked
// form, and the gated RMSNorm + int8 quantization (gated_norm_q_kernel). The state is only read; writes qk / v / gb
// as gdn_prep does (gdn_commit reads them).
// With G_t = exp(g_1 + .. + g_t), a_t = S0 k_t, b_t = S0 q_t, r_t = v_t - G_t a_t:
//   delta = M r, M = (I + B L)^-1 B, L[t][s] = G_t / G_s (k_s . k_t) for s < t, B = diag(beta);
//   o = G b + N r, N = Wkq^T M, Wkq[s][t] = G_t / G_s (k_s . q_t) for s <= t.
// grid 48 (one block per value head), block 512: wave 0 builds M and N (16x16, from the Gram rows) while waves 1..7
// compute a and b for the 8 tiles of 16 value columns on fp16 MFMA (hi + lo parts: ~fp32 accuracy); then o on MFMA.
// Rows t >= n are zero. The row order enters only through the mask anc[t] (bit s: row s precedes row t): a chain
// today; a tree of drafts would use the ancestor sets (and path sums of g).
#define FW 8
#define LP (DK + 4)   // padded LDS row (floats)
__global__ __launch_bounds__(64 * FW) void gdn_fused_kernel(int n, const bf16* __restrict__ proj, int pstride,
                                                            const bf16* __restrict__ hist, const bf16* __restrict__ cw,
                                                            const float* __restrict__ aneg, const float* __restrict__ dtb,
                                                            const float* __restrict__ state, const bf16* __restrict__ gw,
                                                            float* __restrict__ qk, float* __restrict__ vout,
                                                            float* __restrict__ gb, signed char* __restrict__ xq,
                                                            float2* __restrict__ xg, bf16* __restrict__ xzp, int ngp) {
  typedef _Float16 f16x4 __attribute__((ext_vector_type(4)));
  const int h = blockIdx.x, tid = threadIdx.x, w = tid / 64, l = tid % 64, kh = h / 3;
  const int r16 = l % 16, k8 = 8 * (l / 16);
  __shared__ __attribute__((aligned(16))) float sy[3][16][LP];   // q, k, v rows (after the norm: as in qk / v)
  __shared__ __attribute__((aligned(16))) _Float16 sh16[4][16][DK + 8];   // q hi, q lo, k hi, k lo
  __shared__ __attribute__((aligned(16))) float wkk[16][16], wkq[16][16];   // [s][t]
  __shared__ float smm[16][17], snn[16][17];   // M = (I + B L)^-1 B, N = Wkq^T M
  __shared__ float sgc[16], sbt[16];
  // every global load first; rows past n - 1 repeat row n - 1.
  // state tiles (16 value columns each, MFMA operand layout): wave 1: tiles 0 and 7, wave w >= 2: tile w - 1
  const int ntile = w == 0 ? 0 : w == 1 ? 2 : 1;
  float xs[2][DK / 32][8];
#pragma unroll
  for (int u = 0; u < 2; u++) {
    if (u < ntile) {
      const int tile = u == 1 ? 7 : w == 1 ? 0 : w - 1;
      const float* srow = state + ((size_t)h * DK + tile * 16 + r16) * DK + k8;
#pragma unroll
      for (int j = 0; j < DK / 32; j++) {
        *(float4*)xs[u][j] = *(const float4*)(srow + 32 * j);
        *(float4*)(xs[u][j] + 4) = *(const float4*)(srow + 32 * j + 4);
      }
    }
  }
  // conv items: i = tid + 512 m -> rows 4 qt .. 4 qt + 3 (qt = i / 384) of channel i % 384 (q, k of head kh; v of h)
  uint32_t xin[3][7];
  uint2 cwv[3];
#pragma unroll
  for (int m = 0; m < 3; m++) {
    const int i = tid + 512 * m, qt = i / 384, ch = i % 384, part = ch / DK;
    const int cc = (part == 0 ? kh : part == 1 ? 16 + kh : 32 + h) * DK + ch % DK;
    cwv[m] = *(const uint2*)(cw + cc * 4);
#pragma unroll
    for (int r = 0; r < 7; r++) {
      const int rr = min(4 * qt + r, n + 2);
      xin[m][r] = rr < 3 ? *(const unsigned short*)(hist + rr * NCH + cc)
                         : *(const unsigned short*)(proj + (size_t)(rr - 3) * pstride + cc);
    }
  }
  float ga = 0.f, gbv = 0.f, gan = 0.f, gdt = 0.f;   // g / beta: threads 0..15 -> row tid
  if (tid < 16) {
    const int t = min(tid, n - 1);
    gbv = bf(proj[(size_t)t * pstride + NCH + VD + h]);
    ga = bf(proj[(size_t)t * pstride + NCH + VD + NH + h]);
    gan = aneg[h];
    gdt = dtb[h];
  }
  const int nt = tid / 32, ne0 = (tid % 32) * 4;   // norm: row tid / 32, 4 elements
  const uint2 zv = *(const uint2*)(proj + (size_t)min(nt, n - 1) * pstride + NCH + h * DK + ne0);
  const uint2 gwv = *(const uint2*)(gw + ne0);
#pragma unroll
  for (int m = 0; m < 3; m++) {
    const int i = tid + 512 * m, qt = i / 384, ch = i % 384, part = ch / DK, ci = ch % DK;
    const float wc[4] = {__uint_as_float(cwv[m].x << 16), __uint_as_float(cwv[m].x & 0xffff0000u),
                         __uint_as_float(cwv[m].y << 16), __uint_as_float(cwv[m].y & 0xffff0000u)};
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int t = 4 * qt + u;
      float y = 0.f;
      if (t < n) {
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) acc += __uint_as_float(xin[m][u + j] << 16) * wc[j];
        y = rbf(__fdividef(acc, 1.f + __expf(-acc)));
      }
      sy[part][t][ci] = y;
    }
  }
  if (tid < 16) {
    const int t = tid;
    float g = 0.f, beta = 0.f;
    if (t < n) {
      const float x = ga + gdt;
      const float sp = x > 20.f ? x : log1pf(expf(x));
      g = gan * sp;
      beta = 1.f / (1.f + expf(-gbv));
      gb[(t * 2 + 0) * NH + h] = g;
      gb[(t * 2 + 1) * NH + h] = beta;
    }
#pragma unroll
    for (int m = 1; m < 16; m <<= 1) {     // inclusive prefix sum over the chain (a tree: sums along the path)
      float u = __shfl_up(g, m, 16);
      if (t >= m) g += u;
    }
    sgc[t] = g;
    sbt[t] = beta;
  }
  __syncthreads();
  // L2 norm of the q and k rows: 32 rows x 16 threads x 8 elements; fp16 hi / lo copies for the MFMAs
  {
    const int row = tid / 16, part = row / 16, t = row % 16, e0 = (tid % 16) * 8;
    float* y = &sy[part][t][e0];
    float s = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e++) s += y[e] * y[e];
#pragma unroll
    for (int m = 8; m > 0; m >>= 1) s += __shfl_xor(s, m);
    float r = rsqrtf(s + 1e-6f);
    if (part == 0) r *= 0.08838834764831845f;
    f16x8 yh, yl;
#pragma unroll
    for (int e = 0; e < 8; e++) {
      y[e] *= r;
      yh[e] = (_Float16)y[e];
      yl[e] = (_Float16)(y[e] - (float)yh[e]);
    }
    *(f16x8*)&sh16[2 * part][t][e0] = yh;
    *(f16x8*)&sh16[2 * part + 1][t][e0] = yl;
    if (t < n && h % 3 == 0) {
      float* dst = qk + (t * 32 + part * 16 + kh) * DK + e0;
      *(float4*)dst = *(float4*)y;
      *(float4*)(dst + 4) = *(float4*)(y + 4);
    }
    const int tv = tid / 32, ev = (tid % 32) * 4;   // v rows (not normalized) to global
    if (tv < n) *(float4*)(vout + (tv * NH + h) * DK + ev) = *(float4*)&sy[2][tv][ev];
  }
  __syncthreads();
  f32x4 at[2], bq[2];   // per tile: at[t][c] = k_t . S_c (C layout, rows t), bq[c][t] = S_c . q_t (rows c)
  if (w == 0) {
    // wave 0: the Gram rows, their weights, X = (I + B L)^-1 (column j = lane % 16, 16 dependent steps),
    // M = X B and N = Wkq^T M. Meanwhile waves 1..7 compute the state products.
    f32x4 gk = {0, 0, 0, 0}, gq = {0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < DK / 32; j++) {
      const f16x8 qh = *(const f16x8*)&sh16[0][r16][32 * j + k8], ql = *(const f16x8*)&sh16[1][r16][32 * j + k8];
      const f16x8 kh_ = *(const f16x8*)&sh16[2][r16][32 * j + k8], kl = *(const f16x8*)&sh16[3][r16][32 * j + k8];
      gk = __builtin_amdgcn_mfma_f32_16x16x32_f16(kh_, kh_, gk, 0, 0, 0);
      gk = __builtin_amdgcn_mfma_f32_16x16x32_f16(kl, kh_, gk, 0, 0, 0);
      gk = __builtin_amdgcn_mfma_f32_16x16x32_f16(kh_, kl, gk, 0, 0, 0);
      gq = __builtin_amdgcn_mfma_f32_16x16x32_f16(kh_, qh, gq, 0, 0, 0);
      gq = __builtin_amdgcn_mfma_f32_16x16x32_f16(kl, qh, gq, 0, 0, 0);
      gq = __builtin_amdgcn_mfma_f32_16x16x32_f16(kh_, ql, gq, 0, 0, 0);
    }
    {
      const int t = r16;
      const uint32_t anc = (1u << t) - 1u;        // chain (a tree: the ancestor set of t)
      const float gt = sgc[t], bt = sbt[t];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int s = 4 * (l / 16) + i;
        const float e = __expf(gt - sgc[s]);
        wkk[s][t] = (anc >> s) & 1 ? e * gk[i] * bt : 0.f;
        wkq[s][t] = ((anc | (1u << t)) >> s) & 1 ? e * gq[i] : 0.f;
      }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    {
      const int j = r16;
      float x[16];
#pragma unroll
      for (int t = 0; t < 16; t++) x[t] = t == j ? 1.f : 0.f;
#pragma unroll
      for (int s = 0; s < 15; s++) {
        float wr[16];
#pragma unroll
        for (int q = 0; q < 4; q++) *(float4*)(wr + 4 * q) = *(const float4*)&wkk[s][4 * q];
#pragma unroll
        for (int t = s + 1; t < 16; t++) x[t] -= wr[t] * x[s];
      }
      if (l < 16) {
        const float bj = sbt[j];
#pragma unroll
        for (int t = 0; t < 16; t++) smm[t][j] = x[t] * bj;
      }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const int t = 4 * e + l / 16, s = r16;
      float acc = 0.f;
#pragma unroll
      for (int u = 0; u < 16; u++) acc += wkq[u][t] * smm[u][s];
      snn[t][s] = acc;
    }
  } else {
#pragma unroll
    for (int u = 0; u < 2; u++) {
      if (u < ntile) {
        f32x4 a = {0, 0, 0, 0}, b = {0, 0, 0, 0};
#pragma unroll
        for (int j = 0; j < DK / 32; j++) {
          f16x8 shi, slo;
#pragma unroll
          for (int i = 0; i < 8; i++) {
            const _Float16 a16 = (_Float16)xs[u][j][i];
            shi[i] = a16;
            slo[i] = (_Float16)(xs[u][j][i] - (float)a16);
          }
          const f16x8 qh = *(const f16x8*)&sh16[0][r16][32 * j + k8], ql = *(const f16x8*)&sh16[1][r16][32 * j + k8];
          const f16x8 kh_ = *(const f16x8*)&sh16[2][r16][32 * j + k8], kl = *(const f16x8*)&sh16[3][r16][32 * j + k8];
          a = __builtin_amdgcn_mfma_f32_16x16x32_f16(kh_, shi, a, 0, 0, 0);
          a = __builtin_amdgcn_mfma_f32_16x16x32_f16(kl, shi, a, 0, 0, 0);
          a = __builtin_amdgcn_mfma_f32_16x16x32_f16(kh_, slo, a, 0, 0, 0);
          b = __builtin_amdgcn_mfma_f32_16x16x32_f16(shi, qh, b, 0, 0, 0);
          b = __builtin_amdgcn_mfma_f32_16x16x32_f16(slo, qh, b, 0, 0, 0);
          b = __builtin_amdgcn_mfma_f32_16x16x32_f16(shi, ql, b, 0, 0, 0);
        }
        at[u] = a;
        bq[u] = b;
      }
    }
  }
  __syncthreads();
  // o = G b + N r, r_t = v_t - G_t a_t: r in the C layout of at (lane: t = 4 (l / 16) + i, c = l % 16) is the A
  // operand (row c, k = t) of a 16x16x16 MFMA; B operand N^T (k = s, column t'); hi + lo for both
  if (w > 0) {
    f16x4 nhi, nlo;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float nv = snn[r16][4 * (l / 16) + i];
      nhi[i] = (_Float16)nv;
      nlo[i] = (_Float16)(nv - (float)nhi[i]);
    }
    const float gt = __expf(sgc[r16]);
#pragma unroll
    for (int u = 0; u < 2; u++) {
      if (u < ntile) {
        const int c0 = 16 * (u == 1 ? 7 : w == 1 ? 0 : w - 1);
        f16x4 rhi, rlo;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const int t = 4 * (l / 16) + i;
          const float rv = sy[2][t][c0 + r16] - __expf(sgc[t]) * at[u][i];
          rhi[i] = (_Float16)rv;
          rlo[i] = (_Float16)(rv - (float)rhi[i]);
        }
        f32x4 oc = {0, 0, 0, 0};
        oc = __builtin_amdgcn_mfma_f32_16x16x16f16(rhi, nhi, oc, 0, 0, 0);
        oc = __builtin_amdgcn_mfma_f32_16x16x16f16(rlo, nhi, oc, 0, 0, 0);
        oc = __builtin_amdgcn_mfma_f32_16x16x16f16(rhi, nlo, oc, 0, 0, 0);
#pragma unroll
        for (int i = 0; i < 4; i++) sy[0][r16][c0 + 4 * (l / 16) + i] = gt * bq[u][i] + oc[i];   // q no longer read
      }
    }
  }
  __syncthreads();
  // gated RMSNorm + int8 per (row, head), as gated_norm_q_kernel
  {
    const int t = nt, e0 = ne0;
    const float zf[4] = {__uint_as_float(zv.x << 16), __uint_as_float(zv.x & 0xffff0000u), __uint_as_float(zv.y << 16),
                         __uint_as_float(zv.y & 0xffff0000u)};
    const float gwf[4] = {__uint_as_float(gwv.x << 16), __uint_as_float(gwv.x & 0xffff0000u),
                          __uint_as_float(gwv.y << 16), __uint_as_float(gwv.y & 0xffff0000u)};
    float x[4], vq[4];
    float s = 0.f;
#pragma unroll
    for (int e = 0; e < 4; e++) { x[e] = rbf(sy[0][t][e0 + e]); s += x[e] * x[e]; }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) s += __shfl_xor(s, m);
    const float r = rsqrtf(s / 128.f + 1e-6f);
    float a = 0.f;
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const float z = zf[e];
      vq[e] = rbf(rbf(rbf(x[e] * r) * gwf[e]) * (z / (1.f + __expf(-z))));
      a = fmaxf(a, fabsf(vq[e]));
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor(a, m));
    const float inv = a > 0.f ? 127.f / a : 0.f;
    int qs = 0, qi[4];
#pragma unroll
    for (int e = 0; e < 4; e++) { qi[e] = (int)rintf(vq[e] * inv); qs += qi[e]; }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) qs += __shfl_xor(qs, m);
    if (t < n) {
      *(char4*)(xq + (size_t)t * VD + h * DK + e0) = make_char4(qi[0], qi[1], qi[2], qi[3]);
      if (tid % 32 == 0) {
        const float sx = a / 127.f;
        xg[(size_t)t * NH + h] = make_float2(sx / 16.f, 0.f);
        xzp[(size_t)t * ngp + h] = __float2bfloat16(sx * (float)qs);
      }
    }
  }
}

// commit: grid (48 * 128 / VB, L). Advances each layer's state by n = *n_dev rows (from the prep outputs).
__global__ __launch_bounds__(VB * NP) void gdn_commit_kernel(const int* __restrict__ n_dev,
                                                           const float* __restrict__ qk, const float* __restrict__ v,
                                                           const float* __restrict__ gb, float* __restrict__ state,
                                                           int Tmax) {
  int layer = blockIdx.y;
  float* st = state + (size_t)layer * NH * DK * DK;
  run_steps(*n_dev, blockIdx.x / (DK / VB), blockIdx.x % (DK / VB), qk + (size_t)layer * Tmax * 32 * DK,
            v + (size_t)layer * Tmax * NH * DK, gb + (size_t)layer * Tmax * 2 * NH, st, nullptr, st);
}

// conv history after n rows: rows n, n+1, n+2 of [history; proj rows]. grid (NCH/256, L), block 256.
__global__ void conv_commit_kernel(const int* __restrict__ n_dev, const bf16* __restrict__ proj, int pstride,
                                   size_t lproj, bf16* __restrict__ hist) {
  int layer = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
  int n = *n_dev;
  proj += layer * lproj;
  hist += (size_t)layer * 3 * NCH;
  float r0 = xrow(proj, pstride, hist, n, c), r1 = xrow(proj, pstride, hist, n + 1, c),
        r2 = xrow(proj, pstride, hist, n + 2, c);
  hist[c] = __float2bfloat16(r0);
  hist[NCH + c] = __float2bfloat16(r1);
  hist[2 * NCH + c] = __float2bfloat16(r2);
}

// prefill: out (T, 6144) bf16 = gated RMSNorm of o (T, 48, 128) fp32 with z from proj. grid (T, 48), block 128.
template <typename TO>
__global__ __launch_bounds__(128) void gated_norm_kernel(const TO* __restrict__ o, const bf16* __restrict__ proj,
                                                         int pstride, const bf16* __restrict__ gw,
                                                         bf16* __restrict__ out) {
  int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
  __shared__ float red[2];
  float x = rbf((float)o[((size_t)t * NH + h) * DK + i]);
  float s = x * x;
  for (int m = 32; m > 0; m >>= 1) s += __shfl_xor(s, m);
  if ((i & 63) == 0) red[i >> 6] = s;
  __syncthreads();
  float r = rsqrtf((red[0] + red[1]) / 128.f + 1e-6f);
  float z = bf(proj[(size_t)t * pstride + NCH + h * DK + i]);
  out[(size_t)t * VD + h * DK + i] = __float2bfloat16(rbf(rbf(x * r) * bf(gw[i])) * (z / (1.f + __expf(-z))));
}

// verify output for the W4A8 out-projection (eng/kernels/w4a8.py activation format): the gated RMSNorm output of
// one (row, head) is one group of 128, quantized to int8 as w4a8.quant_x does (from the bf16-rounded values).
// xq (T, 6144) int8, xg (T, 48) float2 (sx / 16, 0), xzp (T, ngp) bf16 = sx * sum(q). grid (T, 48), block 128.
__global__ __launch_bounds__(128) void gated_norm_q_kernel(const float* __restrict__ o, const bf16* __restrict__ proj,
                                                           int pstride, const bf16* __restrict__ gw,
                                                           signed char* __restrict__ xq, float2* __restrict__ xg,
                                                           bf16* __restrict__ xzp, int ngp) {
  int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
  __shared__ float red[2];
  __shared__ int reds[2];
  float x = rbf(o[((size_t)t * NH + h) * DK + i]);
  float s = x * x;
  for (int m = 32; m > 0; m >>= 1) s += __shfl_xor(s, m);
  if ((i & 63) == 0) red[i >> 6] = s;
  __syncthreads();
  float r = rsqrtf((red[0] + red[1]) / 128.f + 1e-6f);
  float z = bf(proj[(size_t)t * pstride + NCH + h * DK + i]);
  float v = rbf(rbf(rbf(x * r) * bf(gw[i])) * (z / (1.f + __expf(-z))));
  float a = fabsf(v);
  for (int m = 32; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor(a, m));
  __syncthreads();
  if ((i & 63) == 0) red[i >> 6] = a;
  __syncthreads();
  a = fmaxf(red[0], red[1]);
  float inv = a > 0.f ? 127.f / a : 0.f;
  int q = (int)rintf(v * inv);
  xq[(size_t)t * VD + h * DK + i] = (signed char)q;
  for (int m = 32; m > 0; m >>= 1) q += __shfl_xor(q, m);
  if ((i & 63) == 0) reds[i >> 6] = q;
  __syncthreads();
  if (i == 0) {
    float sx = a / 127.f;
    xg[(size_t)t * NH + h] = make_float2(sx / 16.f, 0.f);
    xzp[(size_t)t * ngp + h] = __float2bfloat16(sx * (float)(reds[0] + reds[1]));
  }
}

// prefill: y (T, 10240) bf16 = silu(conv) of q|k|v (raw, not normalized); gb (T, 2, 48) = g, beta.
// grid (ceil(T / CT), 11), block 128: blockIdx.y < 10: each thread runs 8 consecutive channels over CT positions
// (all CT + 3 input rows loaded first, 16-byte loads and stores); blockIdx.y == 10: g and beta.
#define CT 16
__global__ __launch_bounds__(128) void gdn_conv_kernel(const bf16* __restrict__ proj, int pstride,
                                                       const bf16* __restrict__ hist, const bf16* __restrict__ cw,
                                                       const float* __restrict__ aneg, const float* __restrict__ dtb,
                                                       bf16* __restrict__ y, float* __restrict__ gb, int T) {
  const int t0 = blockIdx.x * CT, i = threadIdx.x;
  const int t1 = min(T, t0 + CT);
  if (blockIdx.y == 10) {
    for (int k = i; k < (t1 - t0) * NH; k += 128) {
      int t = t0 + k / NH, h = k % NH;
      float b = bf(proj[(size_t)t * pstride + NCH + VD + h]);
      float a = bf(proj[(size_t)t * pstride + NCH + VD + NH + h]);
      float x = a + dtb[h];
      float sp = x > 20.f ? x : log1pf(expf(x));
      gb[(t * 2 + 0) * NH + h] = aneg[h] * sp;
      gb[(t * 2 + 1) * NH + h] = 1.f / (1.f + expf(-b));
    }
    return;
  }
  const int c = blockIdx.y * 1024 + i * 8;
  uint4 in[CT + 3];
#pragma unroll
  for (int r = 0; r < CT + 3; r++) {
    const int t = t0 + r - 3;   // input row r is position t0 + r - 3 (history rows for t < 0)
    if (t < t1) in[r] = t < 0 ? *(const uint4*)(hist + (t + 3) * NCH + c) : *(const uint4*)(proj + (size_t)t * pstride + c);
  }
  float w[8][4];
  {
    const uint4* wp = (const uint4*)(cw + (size_t)c * 4);   // 8 channels x 4 taps = 64 bytes
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const uint4 u = wp[j];
      const uint32_t uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int e = 0; e < 4; e++) {
        w[2 * j + e / 2][2 * (e & 1)] = __uint_as_float(uu[e] << 16);
        w[2 * j + e / 2][2 * (e & 1) + 1] = __uint_as_float(uu[e] & 0xffff0000u);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < CT; r++) {
    const int t = t0 + r;
    if (t >= t1) break;
    uint32_t o[4];
#pragma unroll
    for (int p = 0; p < 4; p++) {
      float res[2];
#pragma unroll
      for (int s = 0; s < 2; s++) {
        const int e = 2 * p + s;
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint4& u = in[r + j];
          const uint32_t v = p == 0 ? u.x : p == 1 ? u.y : p == 2 ? u.z : u.w;
          const float x = s == 0 ? __uint_as_float(v << 16) : __uint_as_float(v & 0xffff0000u);
          acc += x * w[e][j];
        }
        res[s] = __fdividef(acc, 1.f + __expf(-acc));
      }
      o[p] = (uint32_t)__bfloat16_as_ushort(__float2bfloat16(res[0])) |
             ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(res[1])) << 16);
    }
    *(uint4*)(y + (size_t)t * NCH + c) = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

extern "C" {
int gdn_conv(int T, const void* proj, int pstride, const void* hist, const void* cw, const void* aneg,
             const void* dtb, void* y, void* gb, hipStream_t s) {
  hipLaunchKernelGGL(gdn_conv_kernel, dim3((T + CT - 1) / CT, 11), dim3(128), 0, s, (const bf16*)proj, pstride,
                     (const bf16*)hist, (const bf16*)cw, (const float*)aneg, (const float*)dtb, (bf16*)y, (float*)gb, T);
  return (int)hipGetLastError();
}
int gated_norm(int T, const void* o, const void* proj, int pstride, const void* gw, void* out, hipStream_t s) {
  hipLaunchKernelGGL(gated_norm_kernel<float>, dim3(T, NH), dim3(128), 0, s, (const float*)o, (const bf16*)proj, pstride,
                     (const bf16*)gw, (bf16*)out);
  return (int)hipGetLastError();
}
int gated_norm_bf16(int T, const void* o, const void* proj, int pstride, const void* gw, void* out, hipStream_t s) {
  hipLaunchKernelGGL(gated_norm_kernel<bf16>, dim3(T, NH), dim3(128), 0, s, (const bf16*)o, (const bf16*)proj,
                     pstride, (const bf16*)gw, (bf16*)out);
  return (int)hipGetLastError();
}
int conv_update(const void* n_dev, const void* proj, int pstride, void* hist, hipStream_t s) {
  hipLaunchKernelGGL(conv_commit_kernel, dim3(NCH / 256, 1), dim3(256), 0, s, (const int*)n_dev, (const bf16*)proj,
                     pstride, (size_t)0, (bf16*)hist);
  return (int)hipGetLastError();
}
int gdn_prep(int T, const void* proj, int pstride, const void* hist, const void* cw, const void* aneg,
             const void* dtb, void* qk, void* v, void* gb, hipStream_t s) {
  hipLaunchKernelGGL(gdn_prep_kernel, dim3(T, 81), dim3(128), 0, s, (const bf16*)proj, pstride, (const bf16*)hist,
                     (const bf16*)cw, (const float*)aneg, (const float*)dtb, (float*)qk, (float*)v, (float*)gb);
  return (int)hipGetLastError();
}
int gdn_verify(int T, const void* qk, const void* v, const void* gb, const void* state, void* o, const void* proj,
               int pstride, const void* gw, void* out, hipStream_t s) {
  hipLaunchKernelGGL(gdn_verify_kernel, dim3(NH, DK / VB), dim3(VB * NP), 0, s, T, (const float*)qk, (const float*)v,
                     (const float*)gb, (const float*)state, (float*)o);
  hipLaunchKernelGGL(gated_norm_kernel<float>, dim3(T, NH), dim3(128), 0, s, (const float*)o, (const bf16*)proj, pstride,
                     (const bf16*)gw, (bf16*)out);
  return (int)hipGetLastError();
}
int gdn_verify_q(int T, const void* qk, const void* v, const void* gb, const void* state, void* o, const void* proj,
                 int pstride, const void* gw, void* xq, void* xg, void* xzp, int ngp, hipStream_t s) {
  hipLaunchKernelGGL(gdn_verify_kernel, dim3(NH, DK / VB), dim3(VB * NP), 0, s, T, (const float*)qk, (const float*)v,
                     (const float*)gb, (const float*)state, (float*)o);
  hipLaunchKernelGGL(gated_norm_q_kernel, dim3(T, NH), dim3(128), 0, s, (const float*)o, (const bf16*)proj, pstride,
                     (const bf16*)gw, (signed char*)xq, (float2*)xg, (bf16*)xzp, ngp);
  return (int)hipGetLastError();
}
int gdn_fused(int T, const void* proj, int pstride, const void* hist, const void* cw, const void* aneg, const void* dtb,
              const void* state, const void* gw, void* qk, void* v, void* gb, void* xq, void* xg, void* xzp, int ngp,
              hipStream_t s) {
  hipLaunchKernelGGL(gdn_fused_kernel, dim3(NH), dim3(64 * FW), 0, s, T, (const bf16*)proj, pstride, (const bf16*)hist,
                     (const bf16*)cw, (const float*)aneg, (const float*)dtb, (const float*)state, (const bf16*)gw,
                     (float*)qk, (float*)v, (float*)gb, (signed char*)xq, (float2*)xg, (bf16*)xzp, ngp);
  return (int)hipGetLastError();
}
int gdn_commit(int L, int Tmax, const void* n_dev, const void* qk, const void* v, const void* gb, void* state,
               const void* proj, int pstride, long lproj, void* hist, hipStream_t s) {
  hipLaunchKernelGGL(gdn_commit_kernel, dim3(NH * DK / VB, L), dim3(VB * NP), 0, s, (const int*)n_dev, (const float*)qk,
                     (const float*)v, (const float*)gb, (float*)state, Tmax);
  hipLaunchKernelGGL(conv_commit_kernel, dim3(NCH / 256, L), dim3(256), 0, s, (const int*)n_dev,
                     (const bf16*)proj, pstride, (size_t)lproj, (bf16*)hist);
  return (int)hipGetLastError();
}
}
