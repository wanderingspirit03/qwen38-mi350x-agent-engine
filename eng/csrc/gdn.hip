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
int gdn_commit(int L, int Tmax, const void* n_dev, const void* qk, const void* v, const void* gb, void* state,
               const void* proj, int pstride, long lproj, void* hist, hipStream_t s) {
  hipLaunchKernelGGL(gdn_commit_kernel, dim3(NH * DK / VB, L), dim3(VB * NP), 0, s, (const int*)n_dev, (const float*)qk,
                     (const float*)v, (const float*)gb, (float*)state, Tmax);
  hipLaunchKernelGGL(conv_commit_kernel, dim3(NCH / 256, L), dim3(256), 0, s, (const int*)n_dev,
                     (const bf16*)proj, pstride, (size_t)lproj, (bf16*)hist);
  return (int)hipGetLastError();
}
}
