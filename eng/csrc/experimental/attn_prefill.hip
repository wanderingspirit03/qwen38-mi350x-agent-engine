// EXPERIMENT, not built or used (slower than the Triton attn_prefill: 1.33-1.5 ms vs 1.18 ms at T=4096, p0=6400;
// see engine-notes/attention.md Part 3). Sweep: python3 -m eng.kernels.tune_prefill_hip "-DKPF=8 -DVPF=4" ...
//
// Prefill attention for the 16 full-attention layers: T new rows of one sequence at positions p0..p0+T-1, causal
// over the keys 0..p0+T-1 already in the cache. 24 query heads, 4 KV heads (head h uses h / 6), head dim 256.
// q (T, 24, 256) bf16 (row strides given), k/v cache (L, 4, 256) bf16; out (T, 24, 256) bf16 (row strides given),
// times sigmoid(gate) when a gate is given.
//
// One workgroup: 128 rows of one head, 4 waves (one per SIMD), 32 rows per wave, blocks of 64 keys.
// K and V go straight from global memory to LDS (global_load_lds), K one block ahead of V:
//   iteration j: loads K(j+2), V(j+1); computes S(j+1) = K(j+1) Q^T with MFMAs interleaved with the softmax
//   VALU work of S(j); then O += P(j) V(j). So the MFMA unit has work while the exponentials run.
// MFMA bf16 16x16x32: S^T tiles (Q^T stays in registers as the B operand), so a lane holds 2 rows x 16 keys;
// P is the A operand of PV straight from registers; V (row-major in LDS) is read with ds_read_b64_tr_b16.
// Online softmax in fp32, base 2; the running max only moves when it grows by more than 8 (P <= 2^8),
// so the O rescale is rare.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef short s16x4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(3))) uint8_t lds_t;

#define PD 256
#define PBM 128
#define PBN 64
#define PKS 1024      // cache elements per position
#define PGROUP 6
#define PRESCALE 8.f  // lazy rescale threshold (log2)
#ifndef KPF
#define KPF 6         // K reads in flight
#endif
#ifndef VPF
#define VPF 3         // V read steps (2 reads each) in flight
#endif
#ifndef SKIP
#define SKIP 0        // timing experiments: 1 no QK MFMA, 2 no PV MFMA, 4 no exp, 8 no global loads
#endif
#ifndef KPERM
#define KPERM 1
#endif
#ifndef SVALU
#define SVALU 4       // VALU instructions per QK MFMA pair in the interleaved part
#endif

// O += A B with the accumulator pinned to AGPRs (the compiler otherwise copies all of O between AGPRs and VGPRs
// every block). The compiler does not see MFMA hazards through asm: reads of acc by VALU need MFMA_DRAIN first.
static __device__ __forceinline__ void mfma_acc(f32x4& acc, bf16x8 a, bf16x8 b) {
  asm("v_mfma_f32_16x16x32_bf16 %0, %1, %2, %0" : "+a"(acc) : "v"(a), "v"(b));
}
#define MFMA_DRAIN() asm volatile("s_nop 7\n\ts_nop 7\n\ts_nop 3" ::: "memory")

static __device__ __forceinline__ float bf2f(uint16_t x) { return __uint_as_float((uint32_t)x << 16); }
static __device__ __forceinline__ uint32_t pk_bf16(float a, float b) {
  return __builtin_bit_cast(uint32_t, __builtin_convertvector((f32x2){a, b}, bf16x2));
}
// max / sum over the 4 lanes lr + 16 lq (lq = 0..3)
static __device__ __forceinline__ float max4(float x) {
  auto a = __builtin_amdgcn_permlane32_swap(__float_as_uint(x), __float_as_uint(x), false, false);
  x = fmaxf(__uint_as_float(a[0]), __uint_as_float(a[1]));
  auto b = __builtin_amdgcn_permlane16_swap(__float_as_uint(x), __float_as_uint(x), false, false);
  return fmaxf(__uint_as_float(b[0]), __uint_as_float(b[1]));
}
static __device__ __forceinline__ float sum4(float x) {
  auto a = __builtin_amdgcn_permlane32_swap(__float_as_uint(x), __float_as_uint(x), false, false);
  x = __uint_as_float(a[0]) + __uint_as_float(a[1]);
  auto b = __builtin_amdgcn_permlane16_swap(__float_as_uint(x), __float_as_uint(x), false, false);
  return __uint_as_float(b[0]) + __uint_as_float(b[1]);
}

// The d chunk (8 values) a lane (lr, lq) holds in QK step s; any bijection works as long as K and Q agree.
// With KPERM, lq goes to chunk bits 3 and 2, so the K reads (chunk ^ lr) are conflict-free for every group of
// 16 lanes made of 16 lr, or 4 lr x 4 lq, or 8 lr x 2 lq.
static __device__ __forceinline__ int kchunk(int s, int lq) {
  return KPERM ? 16 * (s >> 2) + 8 * (lq & 1) + 4 * (lq >> 1) + (s & 3) : 4 * s + lq;
}

// LDS layout of a block (64 keys x 32 chunks of 16 bytes): chunk c of key at slot key * 32 + (c ^ swz(key)).
static __device__ __forceinline__ int kswz(int key) { return key & 15; }
static __device__ __forceinline__ int vswz(int key) { return (key & 7) << 1; }

// 2048 chunks per block, 256 threads: 8 global_load_lds per thread; wave w, load i writes slots (8 w + i) * 64 + lane.
template <bool ISV>
static __device__ __forceinline__ void load_block(const uint16_t* __restrict__ src, lds_t* dst, int b, int n, int wave,
                                                  int lane) {
  if (SKIP & 8) return;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int slot = (8 * wave + i) * 64 + lane;
    const int key = slot >> 5, cpos = slot & 31;
    const uint16_t* s = src + (size_t)min(b + key, n - 1) * PKS + 8 * (cpos ^ (ISV ? vswz(key) : kswz(key)));
    __builtin_amdgcn_global_load_lds((__attribute__((address_space(1))) void*)const_cast<uint16_t*>(s),
                                     (__attribute__((address_space(3))) void*)(dst + (8 * wave + i) * 1024), 16, 0, 0);
  }
}

__shared__ __attribute__((aligned(16))) uint8_t pk0[32768], pk1[32768], pv0[32768], pv1[32768];

__global__ __launch_bounds__(256, 1) void attn_prefill_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
    uint16_t* __restrict__ o, const uint16_t* __restrict__ gate, int p0, int T, int nblk,
    int sq_t, int sq_h, int so_t, int so_h, int sg_t, int sg_h) {
  const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
  const int lr = lane & 15, lq = lane >> 4;
  // 48 workgroups per pair of row blocks, longest rows first; XCD x (= pid % 8) gets the 6 heads of KV head x % 4.
  const int pid = blockIdx.x;
  const int x8 = pid & 7;
  const int h = (x8 & 3) * PGROUP + (pid >> 3) % PGROUP;
  const int r = (pid / (8 * PGROUP)) * 2 + (x8 >> 2);
  if (r >= nblk) return;
  const int start = (nblk - 1 - r) * PBM;
  const int g = h / PGROUP;
  const int n = p0 + T;
  const uint16_t* kg = kc + g * PD;
  const uint16_t* vg = vc + g * PD;
  const int row0 = start + 32 * wave;

  const int nfull = (p0 + start + 1) / PBN;                  // blocks every row of the workgroup sees in full
  const int nb = (min(p0 + start + PBM, n) + PBN - 1) / PBN;  // blocks of the workgroup
  const int nbw = min(nb, (p0 + row0 + 31) / PBN + 1);       // blocks this wave's rows see
  load_block<false>(kg, (lds_t*)pk0, 0, n, wave, lane);
  load_block<true>(vg, (lds_t*)pv0, 0, n, wave, lane);
  if (nb > 1) load_block<false>(kg, (lds_t*)pk1, PBN, n, wave, lane);

  // Q^T as the B operand: lane (lr, lq), step s: q[row 16 x + lr][8 kchunk(s, lq) .. +7]
  bf16x8 qb[2][8];
#pragma unroll
  for (int x = 0; x < 2; x++) {
    const uint16_t* qp = q + (size_t)min(row0 + 16 * x + lr, T - 1) * sq_t + (size_t)h * sq_h;
#pragma unroll
    for (int s = 0; s < 8; s++) qb[x][s] = __builtin_bit_cast(bf16x8, *(const u32x4*)(qp + 8 * kchunk(s, lq)));
  }
  // LDS byte offsets of this lane's reads, as a few per-lane bases plus constants:
  // K read (m, s): key 16 m + lr, chunk kchunk(s, lq) ^ lr  ->  kb[s & 3] + 8192 m + 256 (s >> 2).
  // V transposed read (ks, dt, half): key 32 ks + 16 half + 4 lq + lr / 4, d 16 dt + 4 (lr & 3)
  //   ->  vb[dt & 7] + 16384 ks + 8192 half + 32 (dt & 8).
  int kb[4], vb[8];
#pragma unroll
  for (int sl = 0; sl < 4; sl++) kb[sl] = lr * 512 + 16 * ((kchunk(sl, lq) & 15) ^ lr);
  {
    const int vk = 4 * lq + (lr >> 2), sp = vk & 7, vd = 4 * (lr & 3);
#pragma unroll
    for (int dl = 0; dl < 8; dl++) vb[dl] = vk * 512 + 32 * (dl ^ sp) + 16 * (vd >> 3) + 2 * (vd & 7);
  }

  const float c = 0.0625f * 1.4426950408889634f;
  float m_i[2] = {-INFINITY, -INFINITY}, l_i[2] = {0.f, 0.f};
  f32x4 oacc[2][16];
#pragma unroll
  for (int x = 0; x < 2; x++)
#pragma unroll
    for (int dt = 0; dt < 16; dt++) oacc[x][dt] = f32x4{0.f, 0.f, 0.f, 0.f};
  f32x4 sc[2][4];   // S^T of the current block, unscaled

  auto kread = [&](const lds_t* kl, int e) {
    if (SKIP & 16) return qb[e & 1][(e >> 1) & 7];
    const int s = e >> 2, m = e & 3;     // e = 4 s + m: consecutive MFMAs use different accumulators
    return *(const bf16x8*)(kl + kb[s & 3] + 8192 * m + 256 * (s >> 2));
  };
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  {
#pragma unroll
    for (int m = 0; m < 4; m++) sc[0][m] = sc[1][m] = f32x4{0.f, 0.f, 0.f, 0.f};
    bf16x8 ka[32];
#pragma unroll
    for (int e = 0; e < KPF; e++) ka[e] = kread((lds_t*)pk0, e);
#pragma unroll
    for (int e = 0; e < 32; e++) {
      if (e + KPF < 32) ka[e + KPF] = kread((lds_t*)pk0, e + KPF);
      __builtin_amdgcn_sched_barrier(0);
      sc[0][e & 3] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(ka[e], qb[0][e >> 2], sc[0][e & 3], 0, 0, 0);
      sc[1][e & 3] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(ka[e], qb[1][e >> 2], sc[1][e & 3], 0, 0, 0);
      __builtin_amdgcn_sched_barrier(0);
    }
  }
  __syncthreads();   // every wave is done with K(0) before K(2) overwrites it

  // kcur: K(j), free now (gets K(j+2)); knext: K(j+1); vcur: V(j); vnext: gets V(j+1)
  f32x4 sn[2][4];   // S^T of the next block
  // sc: S(j) (input), sn: S(j + 1) (output); the two unrolled calls swap them
  auto iter = [&](int j, lds_t* kcur, const lds_t* knext, const lds_t* vcur, lds_t* vnext, f32x4 (&sc)[2][4],
                  f32x4 (&sn)[2][4]) {
    const int b = j * PBN;
    if (j + 2 < nb) load_block<false>(kg, kcur, b + 2 * PBN, n, wave, lane);
    if (j + 1 < nb) load_block<true>(vg, vnext, b + PBN, n, wave, lane);
    if (j < nbw) {
      if (j >= nfull) {
#pragma unroll
        for (int x = 0; x < 2; x++) {
          const int pos = p0 + row0 + 16 * x + lr;
#pragma unroll
          for (int m = 0; m < 4; m++)
#pragma unroll
            for (int i = 0; i < 4; i++)
              if (b + 16 * m + 4 * lq + i > pos) sc[x][m][i] = -INFINITY;
        }
      }
      float mnew[2];
      bool need = false;
#pragma unroll
      for (int x = 0; x < 2; x++) {
        float mx = -INFINITY;
#pragma unroll
        for (int m = 0; m < 4; m++) mx = fmaxf(mx, fmaxf(fmaxf(sc[x][m][0], sc[x][m][1]), fmaxf(sc[x][m][2], sc[x][m][3])));
        mnew[x] = fmaxf(m_i[x], max4(mx) * c);
        need |= mnew[x] > m_i[x] + PRESCALE;
      }
      if (__builtin_amdgcn_ballot_w64(need)) {
        MFMA_DRAIN();
#pragma unroll
        for (int x = 0; x < 2; x++) {
          const float alpha = __builtin_amdgcn_exp2f(m_i[x] - mnew[x]);
          m_i[x] = mnew[x];
          l_i[x] *= alpha;
          float ar[4];
#pragma unroll
          for (int i = 0; i < 4; i++) ar[i] = __shfl(alpha, 4 * lq + i);
#pragma unroll
          for (int dt = 0; dt < 16; dt++)
#pragma unroll
            for (int i = 0; i < 4; i++) oacc[x][dt][i] *= ar[i];
        }
      }
      // S(j+1) = K(j+1) Q^T (computed even when this wave does not need block j+1), interleaved with P(j)
#pragma unroll
      for (int m = 0; m < 4; m++) sn[0][m] = sn[1][m] = f32x4{0.f, 0.f, 0.f, 0.f};
      bf16x8 ka[32];
#pragma unroll
      for (int e = 0; e < 32; e++) {
        ka[e] = kread(knext, e);
        if (SKIP & 1) {
          asm volatile("" ::"v"(ka[e]));
          continue;
        }
        sn[0][e & 3] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(ka[e], qb[0][e >> 2], sn[0][e & 3], 0, 0, 0);
        sn[1][e & 3] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(ka[e], qb[1][e >> 2], sn[1][e & 3], 0, 0, 0);
      }
      // P (bf16) as the A operand of k-step ks: keys 32 ks + 4 lq + (0..3) and 32 ks + 16 + 4 lq + (0..3)
      bf16x8 pa[2][2];
#pragma unroll
      for (int x = 0; x < 2; x++) {
        const float mo = m_i[x];
        float ls = 0.f;
        uint32_t pw[8];
#pragma unroll
        for (int m = 0; m < 4; m++) {
          float pv[4];
#pragma unroll
          for (int i = 0; i < 4; i++) {
            pv[i] = (SKIP & 4) ? __builtin_fmaf(sc[x][m][i], c, -mo) : __builtin_amdgcn_exp2f(__builtin_fmaf(sc[x][m][i], c, -mo));
            ls += pv[i];
          }
          pw[2 * m] = pk_bf16(pv[0], pv[1]);
          pw[2 * m + 1] = pk_bf16(pv[2], pv[3]);
        }
        l_i[x] += ls;
        pa[x][0] = __builtin_bit_cast(bf16x8, u32x4{pw[0], pw[1], pw[2], pw[3]});
        pa[x][1] = __builtin_bit_cast(bf16x8, u32x4{pw[4], pw[5], pw[6], pw[7]});
      }
      __builtin_amdgcn_sched_group_barrier(0x100, KPF, 0);
#pragma unroll
      for (int e = 0; e < 32; e++) {
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
        __builtin_amdgcn_sched_group_barrier(0x002, SVALU, 0);
        if (e + KPF < 32) __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
      }
      __builtin_amdgcn_sched_barrier(0);
      // O += P(j) V(j): step e (dt, ks) needs 2 transposed V reads; reads run VPF steps ahead
      s16x4 vr[64];
      auto vfetch = [&](int e) {
        const int dt = e & 15, ks = e >> 4;   // e = 16 ks + dt: consecutive MFMAs use different accumulators
        const lds_t* va = vcur + vb[dt & 7] + 16384 * ks + 32 * (dt & 8);
        if (SKIP & 32) {
          vr[2 * e] = __builtin_bit_cast(s16x4, (u32x2){(uint32_t)vb[e & 7], (uint32_t)e});
          vr[2 * e + 1] = vr[2 * e];
          return;
        }
        vr[2 * e] = __builtin_amdgcn_ds_read_tr16_b64_v4i16((__attribute__((address_space(3))) s16x4*)va);
        vr[2 * e + 1] = __builtin_amdgcn_ds_read_tr16_b64_v4i16((__attribute__((address_space(3))) s16x4*)(va + 8192));
      };
#pragma unroll
      for (int e = 0; e < VPF; e++) vfetch(e);
      asm volatile("s_nop 4" ::: "memory");   // P was written by VALU; asm MFMAs read it
#pragma unroll
      for (int e = 0; e < 32; e++) {
        if (e + VPF < 32) vfetch(e + VPF);
        __builtin_amdgcn_sched_barrier(0);
        const bf16x8 vv = __builtin_bit_cast(bf16x8, __builtin_shufflevector(vr[2 * e], vr[2 * e + 1], 0, 1, 2, 3, 4, 5, 6, 7));
        if (SKIP & 2) {
          asm volatile("" ::"v"(vv), "v"(pa[0][e >> 4]), "v"(pa[1][e >> 4]));
        } else {
          mfma_acc(oacc[0][e & 15], pa[0][e >> 4], vv);
          mfma_acc(oacc[1][e & 15], pa[1][e >> 4], vv);
        }
        __builtin_amdgcn_sched_barrier(0);
      }
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
  };
  for (int j = 0; j < nb; j += 2) {
    iter(j, (lds_t*)pk0, (const lds_t*)pk1, (const lds_t*)pv0, (lds_t*)pv1, sc, sn);
    if (j + 1 < nb) iter(j + 1, (lds_t*)pk1, (const lds_t*)pk0, (const lds_t*)pv1, (lds_t*)pv0, sn, sc);
  }

  MFMA_DRAIN();
  // O rows 16 x + 4 lq + i, d 16 dt + lr
#pragma unroll
  for (int x = 0; x < 2; x++) {
    const float l = sum4(l_i[x]);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float inv = 1.f / __shfl(l, 4 * lq + i);
      const int t = row0 + 16 * x + 4 * lq + i;
      if (t >= T) continue;
      uint16_t* op = o + (size_t)t * so_t + (size_t)h * so_h + lr;
      const uint16_t* gp = gate ? gate + (size_t)t * sg_t + (size_t)h * sg_h + lr : nullptr;
#pragma unroll
      for (int dt = 0; dt < 16; dt++) {
        float v = oacc[x][dt][i] * inv;
        if (gp) v *= 1.f / (1.f + __expf(-bf2f(gp[16 * dt])));
        op[16 * dt] = __builtin_bit_cast(uint16_t, (__bf16)v);
      }
    }
  }
}

extern "C" int attn_prefill(const void* q, const void* k, const void* v, void* o, const void* gate, int p0, int T,
                            int sq_t, int sq_h, int so_t, int so_h, int sg_t, int sg_h, hipStream_t stream) {
  const int nblk = (T + PBM - 1) / PBM;
  const int grid = (nblk + 1) / 2 * 48;
  hipLaunchKernelGGL(attn_prefill_kernel, dim3(grid), dim3(256), 0, stream, (const uint16_t*)q, (const uint16_t*)k,
                     (const uint16_t*)v, (uint16_t*)o, (const uint16_t*)gate, p0, T, nblk, sq_t, sq_h, so_t, so_h,
                     sg_t, sg_h);
  return (int)hipGetLastError();
}
