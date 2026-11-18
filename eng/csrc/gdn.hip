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
#define KP 32
#define VB 32   // value columns per block (block = VB * 4 threads)
static __device__ __forceinline__ float dpp_xor1(float x) {
  return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), 0xB1, 0xF, 0xF, true));
}
static __device__ __forceinline__ float dpp_xor2(float x) {
  return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), 0x4E, 0xF, 0xF, true));
}
static __device__ __forceinline__ float sum4(float x) {
  x += dpp_xor1(x);
  x += dpp_xor2(x);
  return x;
}

static __device__ __forceinline__ float delta_step(float* S, const float* q, const float* k, float vv, float g,
                                                   float beta) {
  float decay = __expf(g);
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
  for (int i = 0; i < KP; i += 4) {
    a0 += S[i] * k[i]; a1 += S[i + 1] * k[i + 1]; a2 += S[i + 2] * k[i + 2]; a3 += S[i + 3] * k[i + 3];
  }
  float kv = sum4((a0 + a1) + (a2 + a3));
  float delta = (vv - decay * kv) * beta;
  a0 = a1 = a2 = a3 = 0.f;
#pragma unroll
  for (int i = 0; i < KP; i += 4) {
    S[i] = S[i] * decay + k[i] * delta; a0 += S[i] * q[i];
    S[i + 1] = S[i + 1] * decay + k[i + 1] * delta; a1 += S[i + 1] * q[i + 1];
    S[i + 2] = S[i + 2] * decay + k[i + 2] * delta; a2 += S[i + 2] * q[i + 2];
    S[i + 3] = S[i + 3] * decay + k[i + 3] * delta; a3 += S[i + 3] * q[i + 3];
  }
  return sum4((a0 + a1) + (a2 + a3));
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
  int tid = threadIdx.x, lc = tid >> 2, part = tid & 3, kh = h / 3, col = vb * VB + lc;
  __shared__ float sq[16][DK], sk[16][DK], sv[16][VB], sg[16], sb[16];
  float S[KP];
  load_state(S, state + ((size_t)h * DK + col) * DK + part * KP);
  for (int i = tid; i < n * DK; i += VB * 4) {
    int t = i >> 7, c = i & 127;
    sq[t][c] = qk[(t * 32 + kh) * DK + c];
    sk[t][c] = qk[(t * 32 + 16 + kh) * DK + c];
  }
  for (int i = tid; i < n * VB; i += VB * 4) {
    int t = i / VB, c = i % VB;
    sv[t][c] = v[(t * NH + h) * DK + vb * VB + c];
  }
  if (tid < n) { sg[tid] = gb[(tid * 2) * NH + h]; sb[tid] = gb[(tid * 2 + 1) * NH + h]; }
  __syncthreads();
  for (int t = 0; t < n; t++) {
    float o = delta_step(S, sq[t] + part * KP, sk[t] + part * KP, sv[t][lc], sg[t], sb[t]);
    if (o_out && part == 0) o_out[((size_t)t * NH + h) * DK + col] = o;
  }
  if (state_out) {
    float4* p4 = reinterpret_cast<float4*>(state_out + ((size_t)h * DK + col) * DK + part * KP);
#pragma unroll
    for (int i = 0; i < KP / 4; i++) p4[i] = make_float4(S[4 * i], S[4 * i + 1], S[4 * i + 2], S[4 * i + 3]);
  }
}

// verify: grid (48, 128 / VB), block VB * 4. state (48, 128 v, 128 k) fp32, read only. o (T, 48, 128) fp32.
__global__ __launch_bounds__(VB * 4) void gdn_verify_kernel(int T, const float* __restrict__ qk,
                                                           const float* __restrict__ v, const float* __restrict__ gb,
                                                           const float* __restrict__ state, float* __restrict__ o) {
  run_steps(T, blockIdx.x, blockIdx.y, qk, v, gb, state, o, nullptr);
}

// commit: grid (48 * 128 / VB, L). Advances each layer's state by n = *n_dev rows (from the prep outputs).
__global__ __launch_bounds__(VB * 4) void gdn_commit_kernel(const int* __restrict__ n_dev,
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
__global__ __launch_bounds__(128) void gated_norm_kernel(const float* __restrict__ o, const bf16* __restrict__ proj,
                                                         int pstride, const bf16* __restrict__ gw,
                                                         bf16* __restrict__ out) {
  int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
  __shared__ float red[2];
  float x = rbf(o[((size_t)t * NH + h) * DK + i]);
  float s = x * x;
  for (int m = 32; m > 0; m >>= 1) s += __shfl_xor(s, m);
  if ((i & 63) == 0) red[i >> 6] = s;
  __syncthreads();
  float r = rsqrtf((red[0] + red[1]) / 128.f + 1e-6f);
  float z = bf(proj[(size_t)t * pstride + NCH + h * DK + i]);
  out[(size_t)t * VD + h * DK + i] = __float2bfloat16(rbf(rbf(x * r) * bf(gw[i])) * (z / (1.f + __expf(-z))));
}

extern "C" {
int gated_norm(int T, const void* o, const void* proj, int pstride, const void* gw, void* out, hipStream_t s) {
  hipLaunchKernelGGL(gated_norm_kernel, dim3(T, NH), dim3(128), 0, s, (const float*)o, (const bf16*)proj, pstride,
                     (const bf16*)gw, (bf16*)out);
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
  hipLaunchKernelGGL(gdn_verify_kernel, dim3(NH, DK / VB), dim3(VB * 4), 0, s, T, (const float*)qk, (const float*)v,
                     (const float*)gb, (const float*)state, (float*)o);
  hipLaunchKernelGGL(gated_norm_kernel, dim3(T, NH), dim3(128), 0, s, (const float*)o, (const bf16*)proj, pstride,
                     (const bf16*)gw, (bf16*)out);
  return (int)hipGetLastError();
}
int gdn_commit(int L, int Tmax, const void* n_dev, const void* qk, const void* v, const void* gb, void* state,
               const void* proj, int pstride, long lproj, void* hist, hipStream_t s) {
  hipLaunchKernelGGL(gdn_commit_kernel, dim3(NH * DK / VB, L), dim3(VB * 4), 0, s, (const int*)n_dev, (const float*)qk,
                     (const float*)v, (const float*)gb, (float*)state, Tmax);
  hipLaunchKernelGGL(conv_commit_kernel, dim3(NCH / 256, L), dim3(256), 0, s, (const int*)n_dev,
                     (const bf16*)proj, pstride, (size_t)lproj, (bf16*)hist);
  return (int)hipGetLastError();
}
}
