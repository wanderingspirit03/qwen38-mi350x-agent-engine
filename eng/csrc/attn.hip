// Decode/verify attention for the 16 full-attention layers: T <= 16 new positions of one sequence.
// 24 query heads, 4 KV heads (query head h uses KV head h / 6), head dim 256, scale 1/16.
// q (T, 24, 256) bf16 (row strides given), k/v cache (L, 4, 256) bf16, keys 0..n-1 valid, n = *n_ptr + n_add.
// Query token t is at position n - T + t and sees keys 0..n-T+t.
//
// Stage 1 (attn_split_kernel): grid (NSPLIT, 4 KV heads), 8 waves. The rows of a workgroup are the 6T
// (token, query head) pairs of its KV head (row r = 6 t + hh), so every K/V byte is read once from HBM.
// Its key range is one of NSPLIT equal chunks of [0, n) (a multiple of 64 long; sized on the device from n).
// It writes O / l (bf16 or fp32) and log2-sum-exp per row; a chunk past n writes lse = -inf.
// Stage 2 (attn_merge_kernel): grid (6T, 4), 256 threads (one per d): combines the chunks, optional gate.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));

#define AD 256        // head dim
#define AGROUP 6      // query heads per KV head
#define AKVH 4
#define AKB 64        // keys per block
#define KSTRIDE 1024  // cache elements per position (4 heads x 256)

static __device__ __forceinline__ float bf2f(uint16_t x) { return __uint_as_float((uint32_t)x << 16); }
static __device__ __forceinline__ uint16_t f2bf(float f) {   // round to nearest even
  uint32_t u = __float_as_uint(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
// Workgroup barrier that waits only for this wave's LDS operations, not for its prefetch loads in flight.
static __device__ __forceinline__ void lds_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
static __device__ __forceinline__ uint32_t pack2(float a, float b) {
  return (uint32_t)f2bf(a) | ((uint32_t)f2bf(b) << 16);
}

// V^T in LDS: quad (4 keys) j of row d stored at d * 16 + (j ^ vswz(d)); conflict-free b64 writes (lanes over
// d = 8o + e, o = 0..15) and reads (lanes over 16 consecutive d).
static __device__ __forceinline__ int vswz(int d) { return (d ^ (d >> 3)) & 15; }

// NW waves (8: two per SIMD, so one wave's latencies overlap the other's work). Per block of 64 keys:
//   1. all waves store the prefetched K and V^T into LDS, then issue the global loads of the next block;
//   2. QK: wave w owns the 16-row tiles w, w + NW, .. (< mt): S^T = K Q^T (Q^T in registers), online softmax
//      (m, l in registers), writes P (bf16, [row][key]) and the row rescale factor alpha into LDS;
//   3. PV: wave w owns d in [DW w, DW w + DW) (DW = 256 / NW) for all rows: O = alpha O + P V.
// MTX: max row tiles (T <= 16 * MTX / 6). NW = 8 (4 was slower at every T).
template <int NW, int MTX, bool PBF16>
__global__ __launch_bounds__(NW * 64) void attn_split_kernel(
    const uint16_t* __restrict__ q, int sq_t, int sq_h, const uint16_t* __restrict__ kc,
    const uint16_t* __restrict__ vc, const int* __restrict__ n_ptr, int n_add, int T, int nsplit,
    void* __restrict__ op, float* __restrict__ lse, int rows_ws, int skip) {
  constexpr int NT = NW * 64;
  constexpr int KU = AKB * 32 / NT;        // 16-byte K units per thread per block
  constexpr int VU = AKB / 4 * 32 / NT;    // V units (4 keys x 8 d) per thread per block
  constexpr int QT = (MTX + NW - 1) / NW;  // row tiles per wave in QK
  constexpr int NDT = AD / 16 / NW;        // 16-wide d tiles per wave in PV
  __shared__ u32x4 klds[AKB * 32];         // [key][16-byte chunk ^ (key & 15)]
  __shared__ u32x2 vlds[AD * 16];          // [d][quad ^ vswz(d)]
  __shared__ u32x2 plds[MTX * 16 * 16];    // [row][quad ^ (row & 15)]  P, bf16
  __shared__ float alds[MTX * 16];         // alpha per row
  __shared__ float mlds[MTX * 16], llds[MTX * 16];

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int wave = tid >> 6;
  const int lr = lane & 15, lq = lane >> 4;
  const int s_id = blockIdx.x, g = blockIdx.y;
  const int R = T * AGROUP;
  const int mt = (R + 15) >> 4;

  // Q^T as B operand: lane (row, lq) step s holds q[row][8 (4 s + lq) .. +7]
  bf16x8 qb[QT][8];
#pragma unroll
  for (int x = 0; x < QT; x++) {
    const int row = (wave + NW * x) * 16 + lr;
    const bool ok = row < R;
    const uint16_t* qp = q + (size_t)(row / AGROUP) * sq_t + (size_t)(g * AGROUP + row % AGROUP) * sq_h;
#pragma unroll
    for (int s = 0; s < 8; s++) {
      u32x4 v = ok ? *(const u32x4*)(qp + 8 * (4 * s + lq)) : u32x4{0, 0, 0, 0};
      qb[x][s] = __builtin_bit_cast(bf16x8, v);
    }
  }

  const int n = *n_ptr + n_add;
  int chunk = (n + nsplit - 1) / nsplit;
  chunk = (chunk + AKB - 1) / AKB * AKB;
  const int lo = s_id * chunk;
  const size_t base = ((size_t)s_id * AKVH + g) * rows_ws;
  if (lo >= n) {
    if (tid < R) lse[base + tid] = -INFINITY;
    return;
  }
  const int hi = min(lo + chunk, n);

  const uint16_t* kg = kc + g * AD;
  const uint16_t* vg = vc + g * AD;
  u32x4 kr[KU], vr[VU][4];
  auto load = [&](int b) {
#pragma unroll
    for (int i = 0; i < KU; i++) {
      int u = tid + i * NT;
      int key = u >> 5, o = u & 31;
      kr[i] = b + key < hi ? *(const u32x4*)(kg + (size_t)(b + key) * KSTRIDE + 8 * o) : u32x4{0, 0, 0, 0};
    }
#pragma unroll
    for (int i = 0; i < VU; i++) {
      int u = tid + i * NT;
      int j = u >> 5, o = u & 31;
#pragma unroll
      for (int c = 0; c < 4; c++) {
        int key = b + 4 * j + c;
        vr[i][c] = key < hi ? *(const u32x4*)(vg + (size_t)key * KSTRIDE + 8 * o) : u32x4{0, 0, 0, 0};
      }
    }
  };
  auto store = [&]() {
#pragma unroll
    for (int i = 0; i < KU; i++) {
      int u = tid + i * NT;
      int key = u >> 5, o = u & 31;
      klds[key * 32 + (o ^ (key & 15))] = kr[i];
    }
#pragma unroll
    for (int i = 0; i < VU; i++) {
      int u = tid + i * NT;
      int j = u >> 5, o = u & 31;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        int d = 8 * o + e;
        int sh = (e & 1) * 16;
        uint32_t x0 = (vr[i][0][e >> 1] >> sh) & 0xFFFFu, x1 = (vr[i][1][e >> 1] >> sh) & 0xFFFFu;
        uint32_t x2 = (vr[i][2][e >> 1] >> sh) & 0xFFFFu, x3 = (vr[i][3][e >> 1] >> sh) & 0xFFFFu;
        vlds[d * 16 + (j ^ vswz(d))] = u32x2{x0 | (x1 << 16), x2 | (x3 << 16)};
      }
    }
  };

  const float sl2 = 0.0625f * 1.4426950408889634f;
  float m_i[QT], l_i[QT];
#pragma unroll
  for (int x = 0; x < QT; x++) {
    m_i[x] = -1e30f;
    l_i[x] = 0.f;
  }
  f32x4 oacc[MTX][NDT];
#pragma unroll
  for (int x = 0; x < MTX; x++)
#pragma unroll
    for (int i = 0; i < NDT; i++) oacc[x][i] = f32x4{0.f, 0.f, 0.f, 0.f};

  load(lo);
  for (int b = lo; b < hi; b += AKB) {
    lds_barrier();
    store();
    if (b + AKB < hi) load(b + AKB);
    lds_barrier();
    if (skip & 1) continue;
#pragma unroll
    for (int x = 0; x < QT; x++) {
      const int tile = wave + NW * x;
      if (tile >= mt) break;
      const int row = tile * 16 + lr;
      const int pos = n - T + row / AGROUP;      // last key this row sees (padded rows see everything)
      // S^T tiles: st[m][i] = S[row][key = b + 16 m + 4 lq + i]
      f32x4 st[4];
#pragma unroll
      for (int m = 0; m < 4; m++) {
        st[m] = f32x4{0.f, 0.f, 0.f, 0.f};
        if (skip & 2) continue;
        int key = 16 * m + lr;
#pragma unroll
        for (int s = 0; s < 8; s++) {
          int c = 4 * s + lq;
          bf16x8 a = __builtin_bit_cast(bf16x8, klds[key * 32 + (c ^ (key & 15))]);
          st[m] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, qb[x][s], st[m], 0, 0, 0);
        }
      }
      float bm = -INFINITY;
#pragma unroll
      for (int m = 0; m < 4; m++)
#pragma unroll
        for (int i = 0; i < 4; i++) {
          int key = b + 16 * m + 4 * lq + i;
          float v = (key < hi && key <= pos) ? st[m][i] * sl2 : -INFINITY;
          st[m][i] = v;
          bm = fmaxf(bm, v);
        }
      bm = fmaxf(bm, __shfl_xor(bm, 16));
      bm = fmaxf(bm, __shfl_xor(bm, 32));
      const float m_new = fmaxf(m_i[x], bm);
      const float alpha = __builtin_amdgcn_exp2f(m_i[x] - m_new);
      m_i[x] = m_new;
      float ls = 0.f;
#pragma unroll
      for (int m = 0; m < 4; m++) {
        f32x4 v = st[m];
        float p0 = __builtin_amdgcn_exp2f(v[0] - m_new), p1 = __builtin_amdgcn_exp2f(v[1] - m_new),
              p2 = __builtin_amdgcn_exp2f(v[2] - m_new), p3 = __builtin_amdgcn_exp2f(v[3] - m_new);
        ls += (p0 + p1) + (p2 + p3);
        plds[row * 16 + ((4 * m + lq) ^ lr)] = u32x2{pack2(p0, p1), pack2(p2, p3)};
      }
      l_i[x] = l_i[x] * alpha + ls;
      if (lq == 0) alds[row] = alpha;
    }
    lds_barrier();
    // PV: O[rows][DW wave + 16 dt + lr]; A = P (lane: row 16 x + lr, keys 32 ks + 8 lq ..), B = V^T
#pragma unroll
    for (int x = 0; x < MTX; x++) {
      if (x >= mt || (skip & 4)) break;
      float ar[4];
#pragma unroll
      for (int i = 0; i < 4; i++) ar[i] = alds[16 * x + 4 * lq + i];
      bf16x8 pa[2];
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        const int row = 16 * x + lr;
        u32x2 p0 = plds[row * 16 + ((8 * ks + 2 * lq) ^ lr)];
        u32x2 p1 = plds[row * 16 + ((8 * ks + 2 * lq + 1) ^ lr)];
        pa[ks] = __builtin_bit_cast(bf16x8, u32x4{p0[0], p0[1], p1[0], p1[1]});
      }
#pragma unroll
      for (int dt = 0; dt < NDT; dt++) {
        const int d = 16 * (NDT * wave + dt) + lr;
        const int sw = vswz(d);
#pragma unroll
        for (int i = 0; i < 4; i++) oacc[x][dt][i] *= ar[i];
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
          u32x2 v0 = vlds[d * 16 + ((8 * ks + 2 * lq) ^ sw)];
          u32x2 v1 = vlds[d * 16 + ((8 * ks + 2 * lq + 1) ^ sw)];
          bf16x8 vb = __builtin_bit_cast(bf16x8, u32x4{v0[0], v0[1], v1[0], v1[1]});
          oacc[x][dt] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(pa[ks], vb, oacc[x][dt], 0, 0, 0);
        }
      }
    }
  }

  // m, l per row -> LDS
#pragma unroll
  for (int x = 0; x < QT; x++) {
    const int tile = wave + NW * x;
    if (tile >= mt) break;
    float l = l_i[x];
    l += __shfl_xor(l, 16);
    l += __shfl_xor(l, 32);
    if (lq == 0) {
      mlds[tile * 16 + lr] = m_i[x];
      llds[tile * 16 + lr] = l;
    }
  }
  lds_barrier();
  if (tid < R) {
    float l = llds[tid];
    lse[base + tid] = l > 0.f ? mlds[tid] + log2f(l) : -INFINITY;
  }
#pragma unroll
  for (int x = 0; x < MTX; x++) {
    if (x >= mt) break;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int r = 16 * x + 4 * lq + i;
      if (r >= R) continue;
      const float l = llds[r];
      const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
      for (int dt = 0; dt < NDT; dt++) {
        float v = oacc[x][dt][i] * inv;
        size_t idx = (base + r) * AD + 16 * (NDT * wave + dt) + lr;
        if (PBF16) ((uint16_t*)op)[idx] = f2bf(v);
        else ((float*)op)[idx] = v;
      }
    }
  }
}

template <bool PBF16>
__global__ __launch_bounds__(256) void attn_merge_kernel(const void* __restrict__ op, const float* __restrict__ lse,
                                                         int nsplit, int rows_ws, uint16_t* __restrict__ o,
                                                         int so_t, int so_h, const uint16_t* __restrict__ gate,
                                                         int sg_t, int sg_h) {
  __shared__ float wsh[256];
  __shared__ float red[4];
  const int r = blockIdx.x, g = blockIdx.y, d = threadIdx.x;
  // thread s < nsplit: weight of chunk s (nsplit <= 256)
  float l = d < nsplit ? lse[((size_t)d * AKVH + g) * rows_ws + r] : -INFINITY;
  float mx = l;
  for (int k = 32; k > 0; k >>= 1) mx = fmaxf(mx, __shfl_xor(mx, k));
  if ((d & 63) == 0) red[d >> 6] = mx;
  __syncthreads();
  mx = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
  float w = exp2f(l - mx);
  wsh[d] = w;
  float ws = w;
  for (int k = 32; k > 0; k >>= 1) ws += __shfl_xor(ws, k);
  __syncthreads();
  if ((d & 63) == 0) red[d >> 6] = ws;
  __syncthreads();
  ws = (red[0] + red[1]) + (red[2] + red[3]);
  float acc = 0.f;
  const size_t step = (size_t)AKVH * rows_ws * AD;
  const size_t i0 = ((size_t)g * rows_ws + r) * AD + d;
#pragma unroll 16
  for (int s = 0; s < nsplit; s++) {
    float x = PBF16 ? bf2f(((const uint16_t*)op)[i0 + s * step]) : ((const float*)op)[i0 + s * step];
    acc += wsh[s] * x;
  }
  acc /= ws;
  const int t = r / AGROUP, h = g * AGROUP + r % AGROUP;
  if (gate) acc *= 1.f / (1.f + __expf(-bf2f(gate[(size_t)t * sg_t + (size_t)h * sg_h + d])));
  o[(size_t)t * so_t + (size_t)h * so_h + d] = f2bf(acc);
}

template <int MTX>
static void launch_split(bool pb, dim3 grid, hipStream_t st, const uint16_t* q, int sq_t, int sq_h,
                         const uint16_t* kc, const uint16_t* vc, const int* n_ptr, int n_add, int T, int nsplit,
                         void* op, float* lse, int rows_ws, int skip) {
  if (pb)
    hipLaunchKernelGGL((attn_split_kernel<8, MTX, true>), grid, dim3(512), 0, st, q, sq_t, sq_h, kc, vc, n_ptr, n_add,
                       T, nsplit, op, lse, rows_ws, skip);
  else
    hipLaunchKernelGGL((attn_split_kernel<8, MTX, false>), grid, dim3(512), 0, st, q, sq_t, sq_h, kc, vc, n_ptr,
                       n_add, T, nsplit, op, lse, rows_ws, skip);
}

// rows_ws: workspace rows per (split, KV head), >= 6T. op: nsplit * 4 * rows_ws * 256 (bf16 if part_bf16 else
// fp32), lse: nsplit * 4 * rows_ws fp32. The partials of unused chunks are never read (weight 0) but must be
// finite: zero op once.
// stages: 3 = both kernels (1 = split only, 2 = merge only; +4: split without the math, +8: without QK,
// +16: without PV; for timing).
extern "C" int attn_decode(int T, const void* q, int sq_t, int sq_h, const void* kc, const void* vc,
                           const int* n_ptr, int n_add, int nsplit, int part_bf16, void* op, float* lse,
                           int rows_ws, void* o, int so_t, int so_h, const void* gate, int sg_t, int sg_h,
                           int stages, hipStream_t st) {
  if (T < 1 || T > 16 || rows_ws < T * AGROUP || nsplit > 256) return -1;
  const int mt = (T * AGROUP + 15) / 16;
  dim3 grid(nsplit, AKVH);
  auto qq = (const uint16_t*)q;
  auto kk = (const uint16_t*)kc;
  auto vv = (const uint16_t*)vc;
  const bool pb = part_bf16 != 0;
  const int skip = (stages >> 2) & 7;
#define ARGS pb, grid, st, qq, sq_t, sq_h, kk, vv, n_ptr, n_add, T, nsplit, op, lse, rows_ws, skip
  if (!(stages & 1)) ;
  else if (mt == 1) launch_split<1>(ARGS);
  else if (mt == 2) launch_split<2>(ARGS);
  else if (mt == 3) launch_split<3>(ARGS);
  else if (mt == 4) launch_split<4>(ARGS);
  else launch_split<6>(ARGS);
#undef ARGS
  dim3 g2(T * AGROUP, AKVH);
  if (!(stages & 2)) ;
  else if (pb)
    hipLaunchKernelGGL(attn_merge_kernel<true>, g2, dim3(256), 0, st, op, lse, nsplit, rows_ws, (uint16_t*)o, so_t,
                       so_h, (const uint16_t*)gate, sg_t, sg_h);
  else
    hipLaunchKernelGGL(attn_merge_kernel<false>, g2, dim3(256), 0, st, op, lse, nsplit, rows_ws, (uint16_t*)o,
                       so_t, so_h, (const uint16_t*)gate, sg_t, sg_h);
  return (int)hipGetLastError();
}
