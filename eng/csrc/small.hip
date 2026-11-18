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

// Row-wise fp8 (e4m3fn) quantization: q = x / s, s = amax(row) / 448. grid R, block 256.
#include <hip/hip_fp8.h>
__global__ __launch_bounds__(256) void quant_fp8_kernel(const bf16* __restrict__ x, int K, int xs,
                                                        __hip_fp8_storage_t* __restrict__ q, float* __restrict__ sc) {
  __shared__ float red[4];
  int r = blockIdx.x;
  const bf16* xr = x + (size_t)r * xs;
  float m = 0.f;
  for (int i = threadIdx.x * 8; i < K; i += 256 * 8) {
    uint4 u = *reinterpret_cast<const uint4*>(xr + i);
    const bf16* e = reinterpret_cast<const bf16*>(&u);
#pragma unroll
    for (int j = 0; j < 8; j++) m = fmaxf(m, fabsf(bf(e[j])));
  }
  for (int o = 32; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o));
  if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = m;
  __syncthreads();
  m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
  float s = fmaxf(m, 1e-8f) / 448.f, inv = 1.f / s;
  if (threadIdx.x == 0) sc[r] = s;
  __hip_fp8_storage_t* qr = q + (size_t)r * K;
  for (int i = threadIdx.x * 8; i < K; i += 256 * 8) {
    uint4 u = *reinterpret_cast<const uint4*>(xr + i);
    const bf16* e = reinterpret_cast<const bf16*>(&u);
    uint2 o;
    __hip_fp8_storage_t* ob = reinterpret_cast<__hip_fp8_storage_t*>(&o);
#pragma unroll
    for (int j = 0; j < 8; j++) ob[j] = __hip_cvt_float_to_fp8(bf(e[j]) * inv, __HIP_SATFINITE, __HIP_E4M3);
    *reinterpret_cast<uint2*>(qr + i) = o;
  }
}

// act = silu(gu[:, :I]) * gu[:, I:] (rounded to bf16 as silu_mul does), quantized to fp8 per row like quant_fp8.
// grid R, block 512; I % 8 == 0, I <= 512 * 8 * 5.
__global__ __launch_bounds__(512) void silu_mul_quant_kernel(const bf16* __restrict__ gu, int I,
                                                              __hip_fp8_storage_t* __restrict__ q,
                                                              float* __restrict__ sc) {
  __shared__ float red[8];
  int r = blockIdx.x, nv = I / 8;
  const bf16* g = gu + (size_t)r * 2 * I;
  float a[5][8];
  float m = 0.f;
#pragma unroll
  for (int k = 0; k < 5; k++) {
    int v = threadIdx.x + k * 512;
    if (v < nv) {
      uint4 ug = *reinterpret_cast<const uint4*>(g + v * 8), uu = *reinterpret_cast<const uint4*>(g + I + v * 8);
      const bf16* eg = reinterpret_cast<const bf16*>(&ug);
      const bf16* eu = reinterpret_cast<const bf16*>(&uu);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        float x = bf(eg[j]);
        a[k][j] = rbf(rbf(x / (1.f + __expf(-x))) * bf(eu[j]));
        m = fmaxf(m, fabsf(a[k][j]));
      }
    }
  }
  for (int o = 32; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o));
  if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = m;
  __syncthreads();
#pragma unroll
  for (int k = 0; k < 8; k++) m = fmaxf(m, red[k]);
  float s = fmaxf(m, 1e-8f) / 448.f, inv = 1.f / s;
  if (threadIdx.x == 0) sc[r] = s;
  __hip_fp8_storage_t* qr = q + (size_t)r * I;
#pragma unroll
  for (int k = 0; k < 5; k++) {
    int v = threadIdx.x + k * 512;
    if (v < nv) {
      uint2 o;
      __hip_fp8_storage_t* ob = reinterpret_cast<__hip_fp8_storage_t*>(&o);
#pragma unroll
      for (int j = 0; j < 8; j++) ob[j] = __hip_cvt_float_to_fp8(a[k][j] * inv, __HIP_SATFINITE, __HIP_E4M3);
      *reinterpret_cast<uint2*>(qr + v * 8) = o;
    }
  }
}

// add_rmsnorm with the normed row quantized to fp8 (as quant_fp8): q (R, H) e4m3, sc (R) row scales.
// grid R, block 128; H = 5120: 5 vectors of 8 per thread.
__global__ __launch_bounds__(128) void add_rmsnorm_q_kernel(bf16* __restrict__ x, const bf16* __restrict__ d,
                                                            const bf16* __restrict__ w,
                                                            __hip_fp8_storage_t* __restrict__ q,
                                                            float* __restrict__ sc, int H) {
  __shared__ float red[2];
  __shared__ float redm[2];
  int r = blockIdx.x;
  x += (size_t)r * H;
  float v[5][8];
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 5; k++) {
    int e = (threadIdx.x + 128 * k) * 8;
    uint4 ux = *reinterpret_cast<const uint4*>(x + e);
    const bf16* bx = reinterpret_cast<const bf16*>(&ux);
    if (d) {
      uint4 ud = *reinterpret_cast<const uint4*>(d + (size_t)r * H + e);
      const bf16* bd = reinterpret_cast<const bf16*>(&ud);
      uint4 uo;
      bf16* bo = reinterpret_cast<bf16*>(&uo);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        v[k][j] = rbf(bf(bx[j]) + bf(bd[j]));
        bo[j] = __float2bfloat16(v[k][j]);
      }
      *reinterpret_cast<uint4*>(x + e) = uo;
    } else {
#pragma unroll
      for (int j = 0; j < 8; j++) v[k][j] = bf(bx[j]);
    }
#pragma unroll
    for (int j = 0; j < 8; j++) s += v[k][j] * v[k][j];
  }
  float tot = block_sum<128>(s, red);
  float rs = rsqrtf(tot / H + 1e-6f);
  float m = 0.f;
#pragma unroll
  for (int k = 0; k < 5; k++) {
    int e = (threadIdx.x + 128 * k) * 8;
    uint4 uw = *reinterpret_cast<const uint4*>(w + e);
    const bf16* bw = reinterpret_cast<const bf16*>(&uw);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      v[k][j] = rbf(v[k][j] * rs * (1.f + bf(bw[j])));
      m = fmaxf(m, fabsf(v[k][j]));
    }
  }
  for (int o = 32; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o));
  if ((threadIdx.x & 63) == 0) redm[threadIdx.x >> 6] = m;
  __syncthreads();
  m = fmaxf(redm[0], redm[1]);
  float sv = fmaxf(m, 1e-8f) / 448.f, inv = 1.f / sv;
  if (threadIdx.x == 0) sc[r] = sv;
#pragma unroll
  for (int k = 0; k < 5; k++) {
    int e = (threadIdx.x + 128 * k) * 8;
    uint2 o;
    __hip_fp8_storage_t* ob = reinterpret_cast<__hip_fp8_storage_t*>(&o);
#pragma unroll
    for (int j = 0; j < 8; j++) ob[j] = __hip_cvt_float_to_fp8(v[k][j] * inv, __HIP_SATFINITE, __HIP_E4M3);
    *reinterpret_cast<uint2*>(q + (size_t)r * H + e) = o;
  }
}

// out[r] = map[argmax_j x[r, j]] (map optional), first index on ties. grid (R, NB), block 256. Partial maxima go to
// ws (R * NB u64); the last block of a row (counter cnt[r]) reduces them and resets the counter.
__device__ __forceinline__ unsigned long long am_key(float v, int j) {
  unsigned int b = __float_as_uint(v);
  b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  return ((unsigned long long)b << 32) | (unsigned int)(0x7fffffff - j);
}
__device__ __forceinline__ unsigned long long am_max(unsigned long long a, unsigned long long b) { return a > b ? a : b; }
__global__ __launch_bounds__(256) void argmax_kernel(const bf16* __restrict__ x, int V, const long long* __restrict__ map,
                                                     long long* __restrict__ out, unsigned long long* __restrict__ ws,
                                                     int* __restrict__ cnt) {
  __shared__ unsigned long long red[4];
  __shared__ int last;
  int r = blockIdx.x, nb = gridDim.y, b = blockIdx.y;
  const bf16* xr = x + (size_t)r * V;
  int per = ((V + nb - 1) / nb + 7) / 8 * 8;
  int j0 = b * per, j1 = min(V, j0 + per);
  unsigned long long best = 0;
  for (int j = j0 + threadIdx.x * 8; j < j1; j += 256 * 8) {
    if (j + 8 <= j1) {
      uint4 u = *reinterpret_cast<const uint4*>(xr + j);
      const bf16* e = reinterpret_cast<const bf16*>(&u);
#pragma unroll
      for (int k = 0; k < 8; k++) best = am_max(best, am_key(bf(e[k]), j + k));
    } else {
      for (int k = j; k < j1; k++) best = am_max(best, am_key(bf(xr[k]), k));
    }
  }
  for (int o = 32; o > 0; o >>= 1) best = am_max(best, (unsigned long long)__shfl_xor((long long)best, o));
  if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = best;
  __syncthreads();
  if (threadIdx.x == 0) {
    best = am_max(am_max(red[0], red[1]), am_max(red[2], red[3]));
    ws[r * nb + b] = best;
    __threadfence();
    last = atomicAdd(cnt + r, 1) == nb - 1;
  }
  __syncthreads();
  if (!last) return;
  if (threadIdx.x == 0) {
    __threadfence();
    unsigned long long m = 0;
    for (int i = 0; i < nb; i++) m = am_max(m, __atomic_load_n(ws + r * nb + i, __ATOMIC_RELAXED));
    int idx = 0x7fffffff - (int)(m & 0xffffffffu);
    out[r] = map ? map[idx] : idx;
    cnt[r] = 0;
  }
}

// MTP input: mx[r, :H] = rmsnorm(embed[tok[r]]) * (1 + we); mx[r, H:] = rmsnorm(hid[r]) * (1 + wh).
// grid (R, 2), block 256, H = 5120.
__global__ __launch_bounds__(256) void mtp_in_kernel(const long long* __restrict__ tok, const bf16* __restrict__ embed,
                                                     const bf16* __restrict__ hid, int hs, const bf16* __restrict__ we,
                                                     const bf16* __restrict__ wh, bf16* __restrict__ mx, int H) {
  __shared__ float red[4];
  int r = blockIdx.x, part = blockIdx.y;
  const bf16* x = part == 0 ? embed + (size_t)tok[r] * H : hid + (size_t)r * hs;
  const bf16* w = part == 0 ? we : wh;
  bf16* out = mx + (size_t)r * 2 * H + part * H;
  float v[20];
  float s = 0.f;
  for (int j = 0; j < 20; j++) {
    int i = threadIdx.x + 256 * j;
    v[j] = i < H ? bf(x[i]) : 0.f;
    s += v[j] * v[j];
  }
  float rs = rsqrtf(block_sum<256>(s, red) / H + 1e-6f);
  for (int j = 0; j < 20; j++) {
    int i = threadIdx.x + 256 * j;
    if (i < H) out[i] = __float2bfloat16(v[j] * rs * (1.f + bf(w[i])));
  }
}

// Acceptance of one speculative round and the position bookkeeping (one block of 64 threads).
// vt (R) = [x, d_1..d_K], y (R) target argmax per row. a = leading matches d_{i+1} == y_i; n = a + 1.
// new[i] = d_{i+1} for i < a, else y_a; res = [n, new...]; vt[0] = y_a; pick = a.
// pos (position of x) advances by n. P (int32): [0] old pos (catch-up), [1] old pos + R (catch-up keys),
// [2] new pos + R (next verify keys), [3 + 2j] new pos + j, [4 + 2j] new pos + j + 1 (chain step j = 0..K-2).
__global__ void accept_kernel(int R, long long* __restrict__ vt, const long long* __restrict__ y,
                              int* __restrict__ n_acc, long long* __restrict__ nw, long long* __restrict__ res,
                              long long* __restrict__ pick, int* __restrict__ pos, int* __restrict__ P) {
  __shared__ int sa;
  int i = threadIdx.x, K = R - 1;
  bool mis = i < K && vt[1 + i] != y[i];
  unsigned long long b = __ballot(mis);
  if (i == 0) sa = b ? __ffsll((long long)b) - 1 : K;
  __syncthreads();
  int a = sa;
  long long ya = y[a];
  long long d = i < K ? vt[1 + i] : 0;
  __syncthreads();
  if (i < R) {
    long long t = i < a ? d : ya;
    nw[i] = t;
    res[1 + i] = t;
  }
  if (i == 0) {
    int p = pos[0], n = a + 1, q = p + n;
    n_acc[0] = n;
    res[0] = n;
    pick[0] = a;
    vt[0] = ya;
    pos[0] = q;
    P[0] = p; P[1] = p + R; P[2] = q + R;
    for (int j = 0; j < K - 1; j++) { P[3 + 2 * j] = q + j; P[4 + 2 * j] = q + j + 1; }
  }
}

extern "C" {
int accept(int R, void* vt, const void* y, void* n_acc, void* nw, void* res, void* pick, void* pos, void* P,
           hipStream_t s) {
  hipLaunchKernelGGL(accept_kernel, dim3(1), dim3(64), 0, s, R, (long long*)vt, (const long long*)y, (int*)n_acc,
                     (long long*)nw, (long long*)res, (long long*)pick, (int*)pos, (int*)P);
  return (int)hipGetLastError();
}
int mtp_in(int R, int H, const void* tok, const void* embed, const void* hid, int hs, const void* we, const void* wh,
           void* mx, hipStream_t s) {
  hipLaunchKernelGGL(mtp_in_kernel, dim3(R, 2), dim3(256), 0, s, (const long long*)tok, (const bf16*)embed,
                     (const bf16*)hid, hs, (const bf16*)we, (const bf16*)wh, (bf16*)mx, H);
  return (int)hipGetLastError();
}
int argmax_rows(int R, int V, int nb, const void* x, const void* map, void* out, void* ws, void* cnt, hipStream_t s) {
  hipLaunchKernelGGL(argmax_kernel, dim3(R, nb), dim3(256), 0, s, (const bf16*)x, V, (const long long*)map,
                     (long long*)out, (unsigned long long*)ws, (int*)cnt);
  return (int)hipGetLastError();
}
int add_rmsnorm_q(int R, int H, void* x, const void* d, const void* w, void* q, void* sc, hipStream_t s) {
  hipLaunchKernelGGL(add_rmsnorm_q_kernel, dim3(R), dim3(128), 0, s, (bf16*)x, (const bf16*)d, (const bf16*)w,
                     (__hip_fp8_storage_t*)q, (float*)sc, H);
  return (int)hipGetLastError();
}
int silu_mul_quant(int R, int I, const void* gu, void* q, void* sc, hipStream_t s) {
  hipLaunchKernelGGL(silu_mul_quant_kernel, dim3(R), dim3(512), 0, s, (const bf16*)gu, I, (__hip_fp8_storage_t*)q,
                     (float*)sc);
  return (int)hipGetLastError();
}
int quant_fp8(int R, int K, const void* x, int xs, void* q, void* sc, hipStream_t s) {
  hipLaunchKernelGGL(quant_fp8_kernel, dim3(R), dim3(256), 0, s, (const bf16*)x, K, xs, (__hip_fp8_storage_t*)q,
                     (float*)sc);
  return (int)hipGetLastError();
}
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
