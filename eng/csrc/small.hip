// Small fused kernels for the decode step (and prefill where shapes allow).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

typedef __hip_bfloat16 bf16;
static __device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }
static __device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16(x)); }

template <int NT>
static __device__ __forceinline__ float block_sum(float s, float* red) {
  for (int o = 32; o > 0; o >>= 1) s += __shfl_xor(s, o);
  int w = threadIdx.x >> 6;
  __syncthreads();
  if ((threadIdx.x & 63) == 0) red[w] = s;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 64; i++) t += red[i];
  return t;
}

// x (R, H) bf16; if d: x += d (bf16 rounding, as the reference's residual adds); out = x/rms(x) * (1 + w).
// grid R, block 256, H = 5120 (20 per thread).
__global__ __launch_bounds__(256) void add_rmsnorm_kernel(bf16* __restrict__ x, const bf16* __restrict__ d,
                                                          const bf16* __restrict__ w, bf16* __restrict__ out, int H,
                                                          int xs, int os) {
  __shared__ float red[4];
  int r = blockIdx.x;
  x += (size_t)r * xs;
  out += (size_t)r * os;
  float v[20];
  float s = 0.f;
  for (int j = 0; j < 20; j++) {
    int i = threadIdx.x + 256 * j;
    float a = 0.f;
    if (i < H) {
      a = bf(x[i]);
      if (d) {
        a = rbf(a + bf(d[(size_t)r * H + i]));
        x[i] = __float2bfloat16(a);
      }
    }
    v[j] = a;
    s += a * a;
  }
  float tot = block_sum<256>(s, red);
  float rs = rsqrtf(tot / H + 1e-6f);
  for (int j = 0; j < 20; j++) {
    int i = threadIdx.x + 256 * j;
    if (i < H) out[i] = __float2bfloat16(v[j] * rs * (1.f + bf(w[i])));
  }
}

// qkv (R, 14336) rows: per query head [q 256 | gate 256] x 24, then k (4 x 256), v (4 x 256).
// grid (R, 32), block 256: y < 24 query heads -> qout (R, 24, 256); 24..27 k heads -> kc[pos+r]; 28..31 v -> vc.
// q and k: RMSNorm over the head (1 + w), then rotary embedding on dims 0..63 (rotate-half pairs (i, i+32)).
__global__ __launch_bounds__(256) void attn_prep_kernel(const bf16* __restrict__ qkv, const bf16* __restrict__ qw,
                                                        const bf16* __restrict__ kw, const int* __restrict__ pos0,
                                                        bf16* __restrict__ qout, bf16* __restrict__ kc,
                                                        bf16* __restrict__ vc) {
  __shared__ float red[4];
  __shared__ float xs[256];
  int r = blockIdx.x, hd = blockIdx.y, i = threadIdx.x;
  int pos = *pos0 + r;
  const bf16* row = qkv + (size_t)r * 14336;
  if (hd >= 28) {
    int kvh = hd - 28;
    vc[((size_t)pos * 4 + kvh) * 256 + i] = row[12288 + 1024 + kvh * 256 + i];
    return;
  }
  bool isq = hd < 24;
  float x = isq ? bf(row[hd * 512 + i]) : bf(row[12288 + (hd - 24) * 256 + i]);
  float tot = block_sum<256>(x * x, red);
  float y = rbf(x * rsqrtf(tot / 256.f + 1e-6f) * (1.f + bf(isq ? qw[i] : kw[i])));
  xs[i] = y;
  __syncthreads();
  if (i < 64) {
    int f = i & 31;
    float inv = powf(10000000.f, -(float)(2 * f) / 64.f);
    float ang = (float)pos * inv;
    float c = cosf(ang), s = sinf(ang);
    float other = xs[i < 32 ? i + 32 : i - 32];
    y = i < 32 ? y * c - other * s : y * c + other * s;
  }
  bf16 o = __float2bfloat16(y);
  if (isq) qout[((size_t)r * 24 + hd) * 256 + i] = o;
  else kc[((size_t)pos * 4 + (hd - 24)) * 256 + i] = o;
}

// act (R, I) = silu(gu[:, :I]) * gu[:, I:]
__global__ void silu_mul_kernel(const bf16* __restrict__ gu, bf16* __restrict__ act, int I, int total) {
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= total) return;
  int r = idx / I, j = idx - r * I;
  float g = bf(gu[(size_t)r * 2 * I + j]), u = bf(gu[(size_t)r * 2 * I + I + j]);
  act[idx] = __float2bfloat16(rbf(g / (1.f + __expf(-g))) * u);
}

// o (R, 6144) *= sigmoid(gate), gate = qkv[r, h*512 + 256 + d]
__global__ void gate_mul_kernel(bf16* __restrict__ o, const bf16* __restrict__ qkv, int total) {
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= total) return;
  int r = idx / 6144, j = idx - r * 6144, h = j >> 8, d = j & 255;
  float g = bf(qkv[(size_t)r * 14336 + h * 512 + 256 + d]);
  o[idx] = __float2bfloat16(bf(o[idx]) * (1.f / (1.f + __expf(-g))));
}

extern "C" {
int add_rmsnorm(int R, int H, void* x, const void* d, const void* w, void* out, int xs, int os, hipStream_t s) {
  hipLaunchKernelGGL(add_rmsnorm_kernel, dim3(R), dim3(256), 0, s, (bf16*)x, (const bf16*)d, (const bf16*)w,
                     (bf16*)out, H, xs, os);
  return (int)hipGetLastError();
}
int attn_prep(int R, const void* qkv, const void* qw, const void* kw, const void* pos0, void* qout, void* kc,
              void* vc, hipStream_t s) {
  hipLaunchKernelGGL(attn_prep_kernel, dim3(R, 32), dim3(256), 0, s, (const bf16*)qkv, (const bf16*)qw,
                     (const bf16*)kw, (const int*)pos0, (bf16*)qout, (bf16*)kc, (bf16*)vc);
  return (int)hipGetLastError();
}
int silu_mul(int R, int I, const void* gu, void* act, hipStream_t s) {
  int total = R * I;
  hipLaunchKernelGGL(silu_mul_kernel, dim3((total + 255) / 256), dim3(256), 0, s, (const bf16*)gu, (bf16*)act, I,
                     total);
  return (int)hipGetLastError();
}
int gate_mul(int R, void* o, const void* qkv, hipStream_t s) {
  int total = R * 6144;
  hipLaunchKernelGGL(gate_mul_kernel, dim3((total + 255) / 256), dim3(256), 0, s, (bf16*)o, (const bf16*)qkv, total);
  return (int)hipGetLastError();
}
}
