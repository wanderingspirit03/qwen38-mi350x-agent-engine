// Chunk-parallel part of the GDN prefill (eng/kernels/gdn_prefill.py; replaces the Triton _wy_kernel): one
// workgroup per (chunk of 64 positions, key head), for the 3 value heads of that key head (they share q, k,
// k k^T and q k^T). Outputs, in the layouts the scan reads:
//   qn (Tp, 16, 128) bf16 = l2norm(q) * scale;  G (48, Tp) fp32 = cumsum of g inside the chunk;
//   M (NC, 48, 64, 64) bf16 = qn_i.kn_j exp(G_i - G_j) for j <= i, else 0;
//   kdT (NC, 48, 128 k, 64 t) bf16 = kn_t exp(G_last - G_t);
//   w (Tp, 48, 128) bf16 = X diag(beta e^G) kn;  uT (NC, 48, 128 v, 64 t) bf16 = (X diag(beta) v)^T,
// with X = (I + A)^-1, A_ij = beta_i kn_i.kn_j exp(G_i - G_j) for j < i.
// X: fp32, one wave per value head (blockwise, see step 3). w and u: bf16 MFMA with the X operand
// X diag(beta e^G) or X diag(beta) rounded to bf16.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __bf16 bf16;
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
#define WMFMA(a, b, c) __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0)

#define WHV 48
#define WD 128
#define WC 64
#define FMFMA(a, b, c) __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0)
// A (and X): the 10 lower 16 x 16 blocks (bi >= bj), each row-major fp32, block (bi, bj) at ablk(bi, bj)
static __host__ __device__ constexpr int ablk(int bi, int bj) { return (bi * (bi + 1) / 2 + bj) * 256; }
constexpr int ASIZE = ablk(4, 0);      // floats per value head

static __device__ __forceinline__ float b2f(uint32_t u, int hi) { return __uint_as_float(hi ? (u & 0xffff0000u) : (u << 16)); }
static __device__ __forceinline__ uint32_t f2b(float x) { return (uint32_t)__builtin_bit_cast(uint16_t, (bf16)x); }
static __device__ __forceinline__ uint32_t pk2(float a, float b) { return f2b(a) | (f2b(b) << 16); }
static __device__ __forceinline__ float lds_bf(const bf16* p) { return __uint_as_float((uint32_t)*(const uint16_t*)p << 16); }

// 64 x 128 bf16 row-major in LDS, 16-byte units XOR-swizzled by row; element offset
static __device__ __forceinline__ int sw128(int row, int col) { return row * 128 + ((((col >> 3) ^ (row & 15))) << 3) + (col & 7); }
// 64 x 64 bf16 (X''), 8 units per row
static __device__ __forceinline__ int sw64(int row, int col) { return row * 64 + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7); }

__global__ __launch_bounds__(256) void gdn_wy_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k,
                                                     const bf16* __restrict__ v, const float* __restrict__ g,
                                                     const float* __restrict__ beta, bf16* __restrict__ w_out,
                                                     bf16* __restrict__ uT_out, bf16* __restrict__ kdT_out,
                                                     float* __restrict__ G_out, bf16* __restrict__ qn_out,
                                                     bf16* __restrict__ M_out, int T, int Tp, long sq, long sk,
                                                     long sv, long sg, long sb, float scale) {
  __shared__ __attribute__((aligned(16))) bf16 skn[64 * 128];
  __shared__ __attribute__((aligned(16))) bf16 sbuf[64 * 128];   // qn, then v of one value head
  __shared__ __attribute__((aligned(16))) float sA[3][ASIZE];    // A, then X'' (bf16) of each value head
  __shared__ float sG[3][64], sB[3][64], sEg[3][64], sEd[3][64];   // G, beta, e^G, e^(G_last - G)
  const int c = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x, lane = tid & 63, wv = tid >> 6;
  const int r = lane & 15, qd = lane >> 4;
  const int t0 = c * WC;

  // ---- 1. normalized k and q -> LDS (q also to global); g cumsum and beta per value head (waves 0..2)
  {
    const int row = tid >> 2, part = tid & 3, t = t0 + row;
    const bool ok = t < T;
#pragma unroll
    for (int which = 0; which < 2; which++) {
      const bf16* src = which ? q : k;
      const long st = which ? sq : sk;
      u32x4 u[4];
      float s = 0.f;
#pragma unroll
      for (int m = 0; m < 4; m++) {
        u[m] = ok ? *(const u32x4*)(src + (size_t)t * st + hk * WD + (part + 4 * m) * 8) : (u32x4){0u, 0u, 0u, 0u};
#pragma unroll
        for (int e = 0; e < 8; e++) { const float x = b2f(u[m][e / 2], e & 1); s += x * x; }
      }
      s += __shfl_xor(s, 1);
      s += __shfl_xor(s, 2);
      const float f = which ? rsqrtf(s + 1e-6f) * scale : rsqrtf(s + 1e-6f);
      bf16* dst = which ? sbuf : skn;
#pragma unroll
      for (int m = 0; m < 4; m++) {
        u32x4 o;
#pragma unroll
        for (int e = 0; e < 4; e++) o[e] = pk2(b2f(u[m][e], 0) * f, b2f(u[m][e], 1) * f);
        const int col = (part + 4 * m) * 8;
        *(u32x4*)(dst + sw128(row, col)) = o;
        if (which) *(u32x4*)(qn_out + ((size_t)t * 16 + hk) * WD + col) = o;
      }
    }
    if (wv < 3) {
      const int h = hk * 3 + wv, t = t0 + lane;
      float x = t < T ? g[(size_t)t * sg + h] : 0.f;
      const float b = t < T ? beta[(size_t)t * sb + h] : 0.f;
#pragma unroll
      for (int d = 1; d < 64; d <<= 1) {
        const float y = __shfl_up(x, d);
        if (lane >= d) x += y;
      }
      sG[wv][lane] = x;
      sEg[wv][lane] = __expf(x);
      sEd[wv][lane] = __expf(__shfl(x, 63) - x);
      sB[wv][lane] = b;
      G_out[(size_t)h * Tp + t] = x;
    }
  }
  __syncthreads();

  // ---- 2. kn kn^T and qn kn^T: row block a = wv, column blocks b <= a. A (blocks, fp32) and M per value head.
  {
    const int a = wv;
    f32x4 kk[4], qk[4];
#pragma unroll
    for (int b = 0; b < 4; b++) {
      kk[b] = (f32x4){0.f, 0.f, 0.f, 0.f};
      qk[b] = (f32x4){0.f, 0.f, 0.f, 0.f};
      if (b > a) continue;
#pragma unroll
      for (int s = 0; s < 4; s++) {
        const int col = 32 * s + 8 * qd;
        const bf16x8 ka = *(const bf16x8*)(skn + sw128(16 * a + r, col));
        const bf16x8 qa = *(const bf16x8*)(sbuf + sw128(16 * a + r, col));
        const bf16x8 kb = *(const bf16x8*)(skn + sw128(16 * b + r, col));
        kk[b] = WMFMA(kb, ka, kk[b]);
        qk[b] = WMFMA(kb, qa, qk[b]);
      }
    }
    // D[j][i]: lane holds i = 16 a + r, j = 16 b + 4 qd + jj
    const int i = 16 * a + r;
#pragma unroll
    for (int e = 0; e < 3; e++) {
      const int h = hk * 3 + e;
      bf16* Mp = M_out + (((size_t)c * WHV + h) * WC + i) * WC;
      const float gi = sG[e][i], bi = sB[e][i];
#pragma unroll
      for (int b = 0; b < 4; b++) {
        float mv[4] = {0.f, 0.f, 0.f, 0.f};
        if (b <= a) {
          f32x4 av;
#pragma unroll
          for (int jj = 0; jj < 4; jj++) {
            const int j = 16 * b + 4 * qd + jj;
            const float d = __expf(gi - sG[e][j]);
            av[jj] = j < i ? kk[b][jj] * bi * d : 0.f;
            mv[jj] = j <= i ? qk[b][jj] * d : 0.f;
          }
          *(f32x4*)(sA[e] + ablk(a, b) + 16 * r + 4 * qd) = av;
        }
        *(u32x2*)(Mp + 16 * b + 4 * qd) = (u32x2){pk2(mv[0], mv[1]), pk2(mv[2], mv[3])};
      }
    }
  }
  __syncthreads();

  // v rows of the first value head -> registers (stored to LDS in step 4; the next head's are loaded meanwhile)
  const int vrow = tid >> 2, vpart = tid & 3;
  u32x4 vr[4];
  auto load_v = [&](int h) {
    const int t = t0 + vrow;
#pragma unroll
    for (int m = 0; m < 4; m++)
      vr[m] = t < T ? *(const u32x4*)(v + (size_t)t * sv + h * WD + (vpart + 4 * m) * 8) : (u32x4){0u, 0u, 0u, 0u};
  };
  load_v(hk * 3);

  // ---- 3. waves 0..2: X = (I + A)^-1 of value head wv, by 16 x 16 blocks: the 4 diagonal blocks by forward
  //         substitution (lane = (block, column)), then the off-diagonal blocks row by row with fp32 MFMA,
  //         X_ij = -X_ii sum_{k=j..i-1} A_ik X_kj, each written over A_ij once A_ij is no longer needed.
  //         Wave 3: kdT of the 3 value heads.
  if (wv < 3) {
    float* Ab = sA[wv];
    {
      float* Ad = Ab + ablk(qd, qd);
      float x[16];
#pragma unroll
      for (int i = 0; i < 16; i++) {
        float a0 = i == r ? 1.f : 0.f, a1 = 0.f;
#pragma unroll
        for (int j4 = 0; j4 < (i + 3) / 4; j4++) {
          const f32x4 av = *(const f32x4*)(Ad + 16 * i + 4 * j4);
#pragma unroll
          for (int jj = 0; jj < 4; jj++) {
            const int j = 4 * j4 + jj;
            if (j < i) {
              if (jj & 1) a1 -= av[jj] * x[j];
              else a0 -= av[jj] * x[j];
            }
          }
        }
        x[i] = a0 + a1;
      }
#pragma unroll
      for (int i = 0; i < 16; i++) Ad[16 * i + r] = x[i];
    }
#pragma unroll
    for (int bi = 1; bi < 4; bi++) {
#pragma unroll
      for (int bj = 0; bj < bi; bj++) {
        f32x4 S = (f32x4){0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int bk = bj; bk < bi; bk++) {
          const f32x4 av = *(const f32x4*)(Ab + ablk(bi, bk) + 16 * r + 4 * qd);
          const float* Xk = Ab + ablk(bk, bj) + 64 * qd + r;
#pragma unroll
          for (int s = 0; s < 4; s++) S = FMFMA(av[s], Xk[16 * s], S);
        }
        const f32x4 xv = *(const f32x4*)(Ab + ablk(bi, bi) + 16 * r + 4 * qd);
        f32x4 P = (f32x4){0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int s = 0; s < 4; s++) P = FMFMA(xv[s], S[s], P);
        float* Xo = Ab + ablk(bi, bj) + 64 * qd + r;
#pragma unroll
        for (int s = 0; s < 4; s++) Xo[16 * s] = -P[s];
      }
    }
  } else {
    // lane: positions 8 t8.. of key dims 8 kg.. (+64 kh); 8 lanes write one 128-byte row of kdT
    const int t8 = lane & 7, kg = lane >> 3;
#pragma unroll 1
    for (int kh = 0; kh < 2; kh++) {
      const int kc0 = 64 * kh + 8 * kg;
      u32x4 rw[8];
#pragma unroll
      for (int e2 = 0; e2 < 8; e2++) rw[e2] = *(const u32x4*)(skn + sw128(8 * t8 + e2, kc0));
#pragma unroll 1
      for (int e = 0; e < 3; e++) {
        float ed[8];
#pragma unroll
        for (int e2 = 0; e2 < 8; e2++) ed[e2] = sEd[e][8 * t8 + e2];
        bf16* dst = kdT_out + (((size_t)c * WHV + hk * 3 + e) * WD + kc0) * WC + 8 * t8;
#pragma unroll
        for (int m = 0; m < 8; m++) {
          u32x4 o;
#pragma unroll
          for (int e2 = 0; e2 < 4; e2++)
            o[e2] = pk2(b2f(rw[2 * e2][m / 2], m & 1) * ed[2 * e2], b2f(rw[2 * e2 + 1][m / 2], m & 1) * ed[2 * e2 + 1]);
          *(u32x4*)(dst + m * WC) = o;
        }
      }
    }
  }
  __syncthreads();

  // ---- 4. per value head: v -> LDS, then w^T = kn^T (X diag(beta e^G))^T and u = (X diag(beta)) v (bf16 MFMA)
  //         the X operands (bf16) are built from the fp32 blocks; the kn^T operands are the same for the 3 heads
  bf16x8 ka[2][2];   // [i2][s]: lane holds kn[32 s + 8 qd + e'][16 nb + r], nb = 2 wv + i2
#pragma unroll
  for (int i2 = 0; i2 < 2; i2++)
#pragma unroll
    for (int s = 0; s < 2; s++) {
      u32x4 ok;
#pragma unroll
      for (int e2 = 0; e2 < 4; e2++) {
        const int j = 32 * s + 8 * qd + 2 * e2, col = 32 * wv + 16 * i2 + r;
        ok[e2] = (uint32_t)*(const uint16_t*)(skn + sw128(j, col)) | ((uint32_t)*(const uint16_t*)(skn + sw128(j + 1, col)) << 16);
      }
      ka[i2][s] = __builtin_bit_cast(bf16x8, ok);
    }
#pragma unroll 1
  for (int e = 0; e < 3; e++) {
    const int h = hk * 3 + e;
#pragma unroll
    for (int m = 0; m < 4; m++) *(u32x4*)(sbuf + sw128(vrow, (vpart + 4 * m) * 8)) = vr[m];
    if (e < 2) load_v(h + 1);
    __syncthreads();
    // xu[tb][s] = X''[t = 16 tb + r][j = 32 s + 8 qd + e'] = X[t][j] beta_j; xw = the same times e^{G_j}
    bf16x8 xu[4][2], xw[4][2];
#pragma unroll
    for (int s = 0; s < 2; s++) {
      float bj[8], ej[8];
#pragma unroll
      for (int e2 = 0; e2 < 8; e2++) {
        bj[e2] = sB[e][32 * s + 8 * qd + e2];
        ej[e2] = bj[e2] * sEg[e][32 * s + 8 * qd + e2];
      }
#pragma unroll
      for (int tb = 2 * s; tb < 4; tb++) {
        const int bjk = 2 * s + (qd >> 1);
        f32x4 x0 = (f32x4){0.f, 0.f, 0.f, 0.f}, x1 = x0;
        if (bjk <= tb) {
          const float* p = sA[e] + ablk(tb, bjk) + 16 * r + 8 * (qd & 1);
          x0 = *(const f32x4*)p;
          x1 = *(const f32x4*)(p + 4);
        }
        u32x4 ou, ow;
#pragma unroll
        for (int e2 = 0; e2 < 2; e2++) {
          ou[e2] = pk2(x0[2 * e2] * bj[2 * e2], x0[2 * e2 + 1] * bj[2 * e2 + 1]);
          ou[e2 + 2] = pk2(x1[2 * e2] * bj[2 * e2 + 4], x1[2 * e2 + 1] * bj[2 * e2 + 5]);
          ow[e2] = pk2(x0[2 * e2] * ej[2 * e2], x0[2 * e2 + 1] * ej[2 * e2 + 1]);
          ow[e2 + 2] = pk2(x1[2 * e2] * ej[2 * e2 + 4], x1[2 * e2 + 1] * ej[2 * e2 + 5]);
        }
        xu[tb][s] = __builtin_bit_cast(bf16x8, ou);
        xw[tb][s] = __builtin_bit_cast(bf16x8, ow);
      }
    }
#pragma unroll
    for (int i2 = 0; i2 < 2; i2++) {
      const int nb = 2 * wv + i2;   // k block (w) and v block (u) of this wave
      bf16x8 vb[2];                 // lane holds v[32 s + 8 qd + e'][16 nb + r]
#pragma unroll
      for (int s = 0; s < 2; s++) {
        u32x4 ov;
#pragma unroll
        for (int e2 = 0; e2 < 4; e2++) {
          const int j = 32 * s + 8 * qd + 2 * e2, col = 16 * nb + r;
          ov[e2] = (uint32_t)*(const uint16_t*)(sbuf + sw128(j, col)) | ((uint32_t)*(const uint16_t*)(sbuf + sw128(j + 1, col)) << 16);
        }
        vb[s] = __builtin_bit_cast(bf16x8, ov);
      }
#pragma unroll
      for (int tb = 0; tb < 4; tb++) {
        f32x4 dw = (f32x4){0.f, 0.f, 0.f, 0.f}, du = (f32x4){0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int s = 0; s < (tb >= 2 ? 2 : 1); s++) {
          dw = WMFMA(ka[i2][s], xw[tb][s], dw);    // D[k][t]
          du = WMFMA(xu[tb][s], vb[s], du);        // D[t][v]
        }
        // w: lane holds k = 16 nb + 4 qd + jj, t = 16 tb + r
        const int tw = t0 + 16 * tb + r;
        *(u32x2*)(w_out + ((size_t)tw * WHV + h) * WD + 16 * nb + 4 * qd) = (u32x2){pk2(dw[0], dw[1]), pk2(dw[2], dw[3])};
        // u: lane holds t = 16 tb + 4 qd + jj, v = 16 nb + r
        *(u32x2*)(uT_out + (((size_t)c * WHV + h) * WD + 16 * nb + r) * WC + 16 * tb + 4 * qd) =
            (u32x2){pk2(du[0], du[1]), pk2(du[2], du[3])};
      }
    }
    __syncthreads();
  }
}
extern "C" int gdn_wy(const void* q, const void* k, const void* v, const void* g, const void* beta, void* w,
                      void* uT, void* kdT, void* G, void* qn, void* M, int T, int Tp, int NC, long sq, long sk,
                      long sv, long sg, long sb, float scale, hipStream_t s) {
  hipLaunchKernelGGL(gdn_wy_kernel, dim3(NC, 16), dim3(256), 0, s, (const bf16*)q, (const bf16*)k, (const bf16*)v,
                     (const float*)g, (const float*)beta, (bf16*)w, (bf16*)uT, (bf16*)kdT, (float*)G, (bf16*)qn,
                     (bf16*)M, T, Tp, sq, sk, sv, sg, sb, scale);
  return (int)hipGetLastError();
}
