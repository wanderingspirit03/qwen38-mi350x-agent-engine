HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef int i32x4 __attribute__((ext_vector_type(4)));

// Packed layout (see w4.py):
//   qw     int32 [N/16][K/128][64 lanes][4]   lane l: row 16t + l%16, k = 128kb + 32s + 8(l/16) + e, dword s,
//                                              element e in nibble e/2 + 4(e%2); stored value u in 0..15
//   meta   uint8 [N/16][K/G][48]                 per (row tile, group): 16 fp16 scales, then 16 bytes 128 + zero point
//          (each workgroup copies the records of its tiles and k range into LDS at the start)
// W[n][k] = scale * (u - z).  MFMA operand value = bf16(128 + u) via magic bits, corrected per group with
// sum_k x[m][k] (computed with an all-ones MFMA).
// MFMA 16x16x32: D[n][m] = sum_k A[n][k] B[k][m]; A = weights (n = row in tile), B = x (m = row of x).
// Lane l holds D[4(l/16) + j][l%16], j = 0..3.
// Lanes whose x row m >= M read row 0 of x and compute unused columns (no branches in the loop).

#ifndef DEQ
#define DEQ deq
#endif

// mask and magic come in registers (magic in a VGPR) so that each pair is one v_and_or_b32
__device__ __forceinline__ bf16x8 deq(uint32_t q, uint32_t mask, uint32_t magic) {
    u32x4 v;
    v[0] = (q & mask) | magic;
    v[1] = ((q >> 4) & mask) | magic;
    v[2] = ((q >> 8) & mask) | magic;
    v[3] = ((q >> 12) & mask) | magic;
    return __builtin_bit_cast(bf16x8, v);
}

template <bool NTL>
__device__ __forceinline__ u32x4 ldw(const u32x4* p) {
#ifndef W4_NO_NT   // experiment switch: plain loads for the weights
    if (NTL) return __builtin_nontemporal_load(p);
#endif
    return *p;
}

#define MFMA __builtin_amdgcn_mfma_f32_16x16x32_bf16
#define BAR asm volatile("" ::: "memory"); __builtin_amdgcn_sched_barrier(0);

template <int NT, int GPB>
struct Frag {
    u32x4 w[NT];
    u32x4 x[4];
};

template <int NT, int GPB>
struct Pend {          // MFMA results of one k block whose scale/zero epilogue is still to be applied
    f32x4 a[NT][GPB];
    f32x4 xs[GPB];
    uint2 sc[NT][GPB];
    uint32_t z[NT][GPB];
};

enum { F_SWIGLU = 1, F_XDUP = 2, F_ARGMAX = 4 };   // F_XDUP: x has K/2 columns, k block kb reads x block kb/2

// F_ARGMAX (M = 1, one k split): each workgroup also does an atomic max of its best key (bf16 value, -n) into one of
// AM_S = 16 u64 slots am[]; a later kernel takes the argmax from the slots and zeroes them (small.hip: am_decode).
// No returning atomics: waiting for one in the epilogue delays every workgroup (+12 us on the draft head).
#define AM_S 16
__device__ __forceinline__ unsigned long long am_key(float v, int j) {
    unsigned int b = __float_as_uint(v);
    b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((unsigned long long)b << 32) | (unsigned int)(0x7fffffff - j);
}

// After the k loop: split-k reduction (the last of the ksplit workgroups of a unit adds the partials up),
// 1/rms scaling, then SwiGLU or the (optional) residual add, and the bf16 store. unit: index of the
// independent group of NT row tiles (counter and sum(x^2) slot). acc[NT] holds x.x^T if RMS.
template <int NT, bool RMS>
__device__ __forceinline__ void finish(f32x4* acc, int unit, int tile0, int lane, __bf16* y, const __bf16* res,
                                       float* ws, int* cnt, int M, int N, int K, int flags, float eps,
                                       unsigned long long* am) {
    const int lg = lane >> 4, m = lane & 15;
    const bool mok = m < M;
    const int ksplit = gridDim.y;
    float ss = 0.f;
    if (RMS) {   // diagonal: D[m][m] sits in lane m + 16 (m / 4), element m % 4
        const int src = m + 16 * (m >> 2);
        const float d0 = __shfl(acc[NT][0], src), d1 = __shfl(acc[NT][1], src);
        const float d2 = __shfl(acc[NT][2], src), d3 = __shfl(acc[NT][3], src);
        const int j = m & 3;
        ss = j == 0 ? d0 : j == 1 ? d1 : j == 2 ? d2 : d3;
    }
    if (ksplit > 1) {
        // write the partial sums, the last workgroup of this row tile adds them up
        float* wss = ws + (size_t)ksplit * M * N + (size_t)unit * ksplit * 16;  // partial sum(x^2)
        if (mok) {
#pragma unroll
            for (int t = 0; t < NT; t++)
                *(f32x4*)(ws + ((size_t)blockIdx.y * M + m) * N + (tile0 + t) * 16 + lg * 4) = acc[t];
            if (RMS && lg == 0) wss[blockIdx.y * 16 + m] = ss;
        }
        __threadfence();
        int prev = 0;
        if (lane == 0) prev = atomicAdd(cnt + unit, 1);
        prev = __shfl(prev, 0);
        if (prev != ksplit - 1) return;
        __threadfence();
        if (lane == 0) cnt[unit] = 0;
        if (!mok) return;
#pragma unroll
        for (int t = 0; t < NT; t++) {
            f32x4 s = f32x4{0.f, 0.f, 0.f, 0.f};
            for (int k = 0; k < ksplit; k++)
                s += __builtin_nontemporal_load((const f32x4*)(ws + ((size_t)k * M + m) * N + (tile0 + t) * 16 + lg * 4));
            acc[t] = s;
        }
        if (RMS) {
            ss = 0.f;
            for (int k = 0; k < ksplit; k++) ss += __builtin_nontemporal_load(wss + k * 16 + m);
        }
    }
    if (!mok) return;
    if (RMS) {
        const float r = rsqrtf(ss / (float)K + eps);
#pragma unroll
        for (int t = 0; t < NT; t++) acc[t] *= r;
    }
    if (flags & F_SWIGLU) {
        const int N2 = N >> 1;
#pragma unroll
        for (int t = 0; t < NT; t += 2) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = ((tile0 + t) >> 1) * 16 + lg * 4 + j;
                const size_t o = (size_t)m * N2 + n;
                const float gt = (float)(__bf16)acc[t][j], up = (float)(__bf16)acc[t + 1][j];
                y[o] = (__bf16)(gt / (1.f + __expf(-gt)) * up);
            }
        }
        return;
    }
    unsigned long long best = 0;
#pragma unroll
    for (int t = 0; t < NT; t++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = (tile0 + t) * 16 + lg * 4 + j;
            const size_t o = (size_t)m * N + n;
            float v = acc[t][j];
            if (res) v += (float)res[o];
            const __bf16 b = (__bf16)v;
            y[o] = b;
            if (flags & F_ARGMAX) {
                const unsigned long long k = am_key((float)b, n);
                best = k > best ? k : best;
            }
        }
    }
    if (flags & F_ARGMAX) {     // lanes 0, 16, 32, 48 (row 0) are the active ones
#pragma unroll
        for (int o = 16; o < 64; o <<= 1) {
            const unsigned long long k = (unsigned long long)__shfl_xor((long long)best, o);
            best = k > best ? k : best;
        }
        if (lane == 0) __hip_atomic_fetch_max(am + blockIdx.x % AM_S, best, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
}

// grid: (N / (16 NT), KSPLIT). Workgroup: WAVES waves on the same NT row tiles, splitting its k range.
// RMS: x is the un-normalized input; the kernel computes sum(x^2) per row (diagonal of an x.x^T MFMA) and
// scales the result by 1/sqrt(mean + eps) (the norm weight is folded into W at quantize time).
// flags & F_SWIGLU: tiles (2i, 2i+1) are (gate, up) rows of the same 16 outputs; y has N/2 columns.
// res may alias y (in-place residual add).
template <int NT, int G, int WAVES, bool NTL, bool RMS>
__global__ __launch_bounds__(WAVES * 64)
void w4_gemm_kernel(const __bf16* x, const u32x4* qw, const u32x4* meta,
                    __bf16* y, const __bf16* res, float* __restrict__ ws, int* __restrict__ cnt,
                    int M, int N, int K, uint32_t mask, uint32_t magic, int flags, float eps,
                    unsigned long long* am) {
    constexpr int GPB = 128 / G;          // groups per 128-k block
    constexpr int SPG = G / 32;           // k-steps (of 32) per group
    constexpr int NA = NT + (RMS ? 1 : 0);
    const int lane = threadIdx.x & 63;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int lg = lane >> 4;
    const int m = lane & 15;
    const int KB = K >> 7;
    const int NG = K / G;
    const int tile0 = blockIdx.x * NT;
    const int ksplit = gridDim.y;
    const int kb0 = (int)(((long)KB * blockIdx.y) / ksplit);
    const int kb1 = (int)(((long)KB * (blockIdx.y + 1)) / ksplit);

    uint32_t magic_v;
    asm volatile("v_mov_b32 %0, %1" : "=v"(magic_v) : "s"(magic));
    magic = magic_v;

    const bf16x8 ones = __builtin_bit_cast(bf16x8, u32x4{0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u});
    f32x4 acc[NA];            // acc[NT]: x.x^T (RMS)
#pragma unroll
    for (int t = 0; t < NA; t++) acc[t] = f32x4{0.f, 0.f, 0.f, 0.f};

    const bool mok = m < M;
    // x through a buffer whose range ends at row M: lanes of rows m >= M read zeros and no memory
    const int xsh = (flags & F_XDUP) ? 1 : 0;
    const int Kx = K >> xsh;
    const __amdgpu_buffer_rsrc_t xrsrc = __builtin_amdgcn_make_buffer_rsrc((void*)x, 0, M * Kx * 2, 0x00020000);
    const int xoff = (m * Kx + lg * 8) * 2;
    const u32x4* qbase = qw + (size_t)tile0 * KB * 64 + lane;
    // meta records of tiles tile0.. for groups [kb0 GPB, kb1 GPB) -> LDS, 3 x 16 bytes per record
    extern __shared__ u32x4 lmeta[];
    const int ngr = (kb1 - kb0) * GPB;
    {
        const int n16 = NT * ngr * 3;
        for (int i = threadIdx.x; i < n16; i += WAVES * 64) {
            const int t = i / (ngr * 3), r = i - t * ngr * 3;
            lmeta[i] = meta[((size_t)(tile0 + t) * NG + kb0 * GPB) * 3 + r];
        }
    }

    auto load = [&](Frag<NT, GPB>& f, int kb) {
        kb = min(kb, kb1 - 1);
#pragma unroll
        for (int t = 0; t < NT; t++) f.w[t] = ldw<NTL>(qbase + ((size_t)t * KB + kb) * 64);
#pragma unroll
        for (int s = 0; s < 4; s++)
            f.x[s] = __builtin_bit_cast(u32x4, __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xoff + (kb >> xsh) * 256 + s * 64, 0, 0));
    };
    const char* lm = (const char*)lmeta;
    auto meta_lds = [&](Pend<NT, GPB>& p, int kb) {
#pragma unroll
        for (int t = 0; t < NT; t++)
#pragma unroll
            for (int g = 0; g < GPB; g++) {
                const char* r = lm + (t * ngr + (kb - kb0) * GPB + g) * 48;
                p.sc[t][g] = *(const uint2*)(r + lg * 8);
                p.z[t][g] = *(const uint32_t*)(r + 32 + lg * 4);
            }
    };
    auto mma = [&](const Frag<NT, GPB>& f, Pend<NT, GPB>& p, int kb) {
        meta_lds(p, kb);
#pragma unroll
        for (int g = 0; g < GPB; g++) {
            f32x4 xs = f32x4{0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int s = 0; s < SPG; s++)
                xs = MFMA(ones, __builtin_bit_cast(bf16x8, f.x[g * SPG + s]), xs, 0, 0, 0);
            p.xs[g] = xs;
#pragma unroll
            for (int t = 0; t < NT; t++) {
                f32x4 a = f32x4{0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int s = 0; s < SPG; s++)
                    a = MFMA(DEQ(f.w[t][g * SPG + s], mask, magic), __builtin_bit_cast(bf16x8, f.x[g * SPG + s]), a, 0, 0, 0);
                p.a[t][g] = a;
            }
        }
        if (RMS) {
#pragma unroll
            for (int s = 0; s < 4; s++)
                acc[NT] = MFMA(__builtin_bit_cast(bf16x8, f.x[s]), __builtin_bit_cast(bf16x8, f.x[s]), acc[NT], 0, 0, 0);
        }
    };
    auto epi = [&](const Pend<NT, GPB>& p) {
#pragma unroll
        for (int g = 0; g < GPB; g++)
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const __half2 s01 = __builtin_bit_cast(__half2, p.sc[t][g].x);
                const __half2 s23 = __builtin_bit_cast(__half2, p.sc[t][g].y);
                const float sj[4] = {__low2float(s01), __high2float(s01), __low2float(s23), __high2float(s23)};
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float c = (float)((p.z[t][g] >> (8 * j)) & 0xFF);
                    acc[t][j] += sj[j] * (p.a[t][g][j] - c * p.xs[g][0]);
                }
            }
    };

    {
        Frag<NT, GPB> fa, fb;
        Pend<NT, GPB> pa, pb;
        int kb = kb0 + wave;
        if (kb < kb1) {
            load(fa, kb);
            load(fb, kb + WAVES);
        }
        __syncthreads();
        if (kb < kb1) {
            BAR
            mma(fa, pa, kb);
            BAR
            while (true) {
                kb += WAVES;
                if (kb >= kb1) { epi(pa); break; }
                load(fa, kb + WAVES);
                BAR
                mma(fb, pb, kb);
                epi(pa);
                BAR
                kb += WAVES;
                if (kb >= kb1) { epi(pb); break; }
                load(fb, kb + WAVES);
                BAR
                mma(fa, pa, kb);
                epi(pb);
                BAR
            }
        }
    }

    if (WAVES > 1) {
        __shared__ f32x4 red[WAVES > 1 ? WAVES - 1 : 1][NA][64];
        if (wave > 0) {
#pragma unroll
            for (int t = 0; t < NA; t++) red[wave - 1][t][lane] = acc[t];
        }
        __syncthreads();
        if (wave != 0) return;
#pragma unroll
        for (int t = 0; t < NA; t++)
            for (int w = 0; w < WAVES - 1; w++) acc[t] += red[w][t][lane];
    }
    finish<NT, RMS>(acc, blockIdx.x, tile0, lane, y, res, ws, cnt, M, N, K, flags, eps, am);
}

// ---- M = 1 GEMV (same packed format; G = 128): the streaming schedule of engine-notes/stream.md.
// Workgroup: NT row tiles over all of K; its W waves take contiguous k ranges [J0, J1). Each wave stages its x
// range (bf16) and its meta records into a wave-private LDS region, issuing those loads BEFORE its first weight
// loads, with the per-block sums of x (zero-point correction) and sum(x^2) (RMS) computed on the way.
// Partial sums are reduced across the waves in LDS; wave 0 runs the epilogue (1/rms, SwiGLU / residual, argmax).
// x is broadcast to all 16 MFMA columns (every lane of a 16-lane group reads the same LDS address).
#if defined(GV_WPE) && GV_WPE > 0
#define GV_ATTR __attribute__((amdgpu_waves_per_eu(GV_WPE)))
#else
#define GV_ATTR
#endif
// I8: x quantized to int8 per 128-k block during staging (scale amax / 127), weights used as unsigned nibbles in
// int8 MFMAs (16x16x64, 2 per tile and block instead of 4 bf16 ones); zero point via the integer sum of x.
// Byte b of (q & 0x0F0F0F0F) holds element (0, 4, 1, 5)[b] of the lane's 8 k values, of (q >> 4) & ...: (2, 6, 3, 7).
template <int NT, int W, int D, int CW, bool RMS, bool I8, int MODE = 0>   // MODE (ablation): 1 no math, 2 no staging either
__global__ __launch_bounds__(W * 64) GV_ATTR
void w4_gemv_kernel(const __bf16* x, const u32x4* __restrict__ qw, const u32x4* __restrict__ meta,
                    __bf16* y, const __bf16* res, int N, int K, uint32_t mask, uint32_t magic, int flags, float eps,
                    unsigned long long* am, float* ws, int* cnt) {
    constexpr int XMAX = (CW * 16 + 63) / 64, MMAX = (NT * CW * 3 + 63) / 64, CWM = CW;
    const int lane = threadIdx.x & 63, lg = lane >> 4;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int KB = K >> 7;
    const int J0 = (blockIdx.y * W + wave) * CW;     // KB = gridDim.y W CW (split k: blockIdx.y)
    const int tile0 = blockIdx.x * NT;
    uint32_t magic_v;
    asm volatile("v_mov_b32 %0, %1" : "=v"(magic_v) : "s"(magic));
    magic = magic_v;

    extern __shared__ char lds[];
    constexpr int XB = I8 ? 128 : 256;                        // x bytes per k block in LDS
    char* wl = lds + wave * CWM * (XB + NT * 48 + 8);
    u32x4* lx = (u32x4*)wl;                                   // bf16: [CW][16] x pieces; I8: [CW][4 lg][lo, hi]
    u32x4* lm = (u32x4*)(wl + CWM * XB);                      // [NT][CW][3] meta records
    float* lxs = (float*)(wl + CWM * (XB + NT * 48));         // [CW] sum of x per k block (I8: of the int8 x)
    float* lgx = lxs + CWM;                                   // [CW] I8: x scale

    // staging loads first
    u32x4 xv[XMAX], mv[MMAX];
#pragma unroll
    for (int k = 0; k < XMAX; k++) {
        const int i = lane + 64 * k;
        xv[k] = MODE != 2 && i < CW * 16 ? *(const u32x4*)(x + (size_t)J0 * 128 + i * 8) : u32x4{0u, 0u, 0u, 0u};
    }
#pragma unroll
    for (int k = 0; k < MMAX; k++) {
        const int i = lane + 64 * k, t = i / (CW * 3), r = i - t * CW * 3;
        if (MODE != 2 && i < NT * CW * 3) mv[k] = meta[((size_t)(tile0 + t) * KB + J0) * 3 + r];
    }
    u32x4 ring[D + 1][NT];   // block i in ring[i % (D + 1)]; loop unrolled by D + 1, so no register copies
    auto load = [&](u32x4* f, int j) {
        j += J0;
#pragma unroll
        for (int t = 0; t < NT; t++) f[t] = __builtin_nontemporal_load(qw + ((size_t)(tile0 + t) * KB + j) * 64 + lane);
    };
#pragma unroll
    for (int d = 0; d < D; d++) load(ring[d], d);
    // residual: a buffer of size 0 when there is none (reads give 0, no branch)
    const __amdgpu_buffer_rsrc_t rr = __builtin_amdgcn_make_buffer_rsrc((void*)res, 0, res ? N * 2 : 0, 0x00020000);
    uint2 rv[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) rv[t] = __builtin_bit_cast(uint2, __builtin_amdgcn_raw_buffer_load_b64(rr, ((tile0 + t) * 16 + lg * 4) * 2, 0, 0));
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < XMAX; k++) {
        const int i = lane + 64 * k;
        const bf16x8 v = __builtin_bit_cast(bf16x8, xv[k]);
        float f[8];
#pragma unroll
        for (int e = 0; e < 8; e++) { f[e] = (float)v[e]; if (RMS) ss += f[e] * f[e]; }
        if constexpr (I8) {
            float mx = 0.f;
#pragma unroll
            for (int e = 0; e < 8; e++) mx = fmaxf(mx, fabsf(f[e]));
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) mx = fmaxf(mx, __shfl_xor(mx, o));
            const float inv = mx > 0.f ? 127.f / mx : 0.f;
            int q[8], s = 0;
#pragma unroll
            for (int e = 0; e < 8; e++) { q[e] = __float2int_rn(f[e] * inv); s += q[e]; }
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) s += __shfl_xor(s, o);
            const uint32_t lo = (q[0] & 255) | (q[4] & 255) << 8 | (q[1] & 255) << 16 | (uint32_t)(q[5] & 255) << 24;
            const uint32_t hi = (q[2] & 255) | (q[6] & 255) << 8 | (q[3] & 255) << 16 | (uint32_t)(q[7] & 255) << 24;
            if (i < CW * 16) {
                const int p = i & 15, kb = i >> 4;     // piece p: k = 8p.. = 32 s + 8 lg
                uint32_t* l32 = (uint32_t*)lx + kb * 32 + (p & 3) * 8 + (p >> 2);
                l32[0] = lo;
                l32[4] = hi;
                if ((lane & 15) == 0) { lxs[kb] = (float)s; lgx[kb] = mx / 127.f; }
            }
        } else {
            float s = 0.f;
#pragma unroll
            for (int e = 0; e < 8; e++) s += f[e];
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) s += __shfl_xor(s, o);
            if (i < CW * 16) {
                lx[i] = xv[k];
                if ((lane & 15) == 0) lxs[i >> 4] = s;
            }
        }
    }
#pragma unroll
    for (int k = 0; k < MMAX; k++) { const int i = lane + 64 * k; if (i < NT * CW * 3) lm[i] = mv[k]; }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

    f32x4 acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = f32x4{0.f, 0.f, 0.f, 0.f};
    const char* lmc = (const char*)lm;
    auto step = [&](const u32x4* f, int j) {
        if constexpr (MODE != 0) {
#pragma unroll
            for (int t = 0; t < NT; t++) acc[t][0] += __uint_as_float((f[t][0] ^ f[t][3]) & 0x3fffffffu);
            return;
        }
        if constexpr (I8) {
            const u32x4 xl = lx[j * 8 + lg * 2], xh = lx[j * 8 + lg * 2 + 1];
            const float xsum = lxs[j], gx = lgx[j];
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const char* r = lmc + (t * CW + j) * 48;
                const uint2 sc = *(const uint2*)(r + lg * 8);
                const uint32_t z = *(const uint32_t*)(r + 32 + lg * 4);
                i32x4 lo, hi;
#pragma unroll
                for (int e = 0; e < 4; e++) { lo[e] = (int)(f[t][e] & 0x0F0F0F0Fu); hi[e] = (int)((f[t][e] >> 4) & 0x0F0F0F0Fu); }
                i32x4 ai = __builtin_amdgcn_mfma_i32_16x16x64_i8(lo, __builtin_bit_cast(i32x4, xl), i32x4{0, 0, 0, 0}, 0, 0, 0);
                ai = __builtin_amdgcn_mfma_i32_16x16x64_i8(hi, __builtin_bit_cast(i32x4, xh), ai, 0, 0, 0);
                const __half2 s01 = __builtin_bit_cast(__half2, sc.x), s23 = __builtin_bit_cast(__half2, sc.y);
                const float sj[4] = {__low2float(s01), __high2float(s01), __low2float(s23), __high2float(s23)};
#pragma unroll
                for (int e = 0; e < 4; e++)
                    acc[t][e] += sj[e] * gx * ((float)ai[e] - (float)((int)((z >> (8 * e)) & 0xFF) - 128) * xsum);
            }
        } else {
            u32x4 xs4[4];
#pragma unroll
            for (int s = 0; s < 4; s++) xs4[s] = lx[j * 16 + s * 4 + lg];
            const float xsum = lxs[j];
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const char* r = lmc + (t * CW + j) * 48;
                const uint2 sc = *(const uint2*)(r + lg * 8);
                const uint32_t z = *(const uint32_t*)(r + 32 + lg * 4);
                f32x4 a = f32x4{0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int s = 0; s < 4; s++) a = MFMA(DEQ(f[t][s], mask, magic), __builtin_bit_cast(bf16x8, xs4[s]), a, 0, 0, 0);
                const __half2 s01 = __builtin_bit_cast(__half2, sc.x), s23 = __builtin_bit_cast(__half2, sc.y);
                const float sj[4] = {__low2float(s01), __high2float(s01), __low2float(s23), __high2float(s23)};
#pragma unroll
                for (int e = 0; e < 4; e++) acc[t][e] += sj[e] * (a[e] - (float)((z >> (8 * e)) & 0xFF) * xsum);
            }
        }
    };
    // ring slot of block j: j % (D + 1). Rolled loop over groups of D + 1 blocks (no cross-group hoisting of the
    // LDS reads, few registers), loads past the end clamped to the last block; then the static tail.
    constexpr int TB = CW - CW % (D + 1);
#pragma unroll 1
    for (int b = 0; b < TB; b += D + 1) {
#pragma unroll
        for (int u = 0; u <= D; u++) {
            BAR
            load(ring[(u + D) % (D + 1)], min(b + u + D, CW - 1));
            BAR
            step(ring[u], b + u);
        }
    }
#pragma unroll
    for (int u = 0; u < CW - TB; u++) {
        BAR
        step(ring[u], TB + u);
    }
    // cross-wave reduction (wave 0 keeps the sums)
    __shared__ f32x4 red[W > 1 ? W - 1 : 1][NT][64];
    __shared__ float rss[W];
    if (RMS) {
#pragma unroll
        for (int o = 1; o < 64; o <<= 1) ss += __shfl_xor(ss, o);
        if (lane == 0) rss[wave] = ss;
    }
    if (W > 1) {
        if (wave > 0) {
#pragma unroll
            for (int t = 0; t < NT; t++) red[wave - 1][t][lane] = acc[t];
        }
        __syncthreads();
        if (wave != 0) return;
#pragma unroll
        for (int t = 0; t < NT; t++)
            for (int w = 0; w < W - 1; w++) acc[t] += red[w][t][lane];
    }
    float s2 = 0.f;
    if (RMS)
        for (int w = 0; w < W; w++) s2 += rss[w];
    const int SK = gridDim.y;
    if (SK > 1) {   // split k: partial sums to ws[split][N]; the last workgroup of the row tiles adds them up.
        // The partials go around the (per-XCD, non-coherent) L2 (sc0 sc1) and the arrival counter is a relaxed
        // device-scope atomic: no release/acquire fences (those write back / invalidate the whole L2).
        float* wss = ws + (size_t)SK * N;   // partial sum(x^2) per (unit, split)
        if ((lane & 15) == 0) {
#pragma unroll
            for (int t = 0; t < NT; t++) {
                float* a = ws + (size_t)blockIdx.y * N + (tile0 + t) * 16 + lg * 4;
                asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1" :: "v"(a), "v"(acc[t]) : "memory");
            }
        }
        if (RMS && lane == 0) {
            float* a = wss + blockIdx.x * SK + blockIdx.y;
            asm volatile("global_store_dword %0, %1, off sc0 sc1" :: "v"(a), "v"(s2) : "memory");
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        int prev = 0;
        if (lane == 0) prev = __hip_atomic_fetch_add(cnt + blockIdx.x, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        prev = __shfl(prev, 0);
        if (prev != SK - 1) return;
        if (lane == 0) __hip_atomic_store(cnt + blockIdx.x, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#pragma unroll
        for (int t = 0; t < NT; t++) {
            f32x4 s = f32x4{0.f, 0.f, 0.f, 0.f};
            for (int k = 0; k < SK; k++) {
                const float* a = ws + (size_t)k * N + (tile0 + t) * 16 + lg * 4;
                f32x4 v;
                asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1\n s_waitcnt vmcnt(0)" : "=v"(v) : "v"(a) : "memory");
                s += v;
            }
            acc[t] = s;
        }
        if (RMS) {
            s2 = 0.f;
            for (int k = 0; k < SK; k++) {
                const float* a = wss + blockIdx.x * SK + k;
                float v;
                asm volatile("global_load_dword %0, %1, off sc0 sc1\n s_waitcnt vmcnt(0)" : "=v"(v) : "v"(a) : "memory");
                s2 += v;
            }
        }
    }
    if ((lane & 15) != 0) return;
    if (RMS) {
        const float r = rsqrtf(s2 / (float)K + eps);
#pragma unroll
        for (int t = 0; t < NT; t++) acc[t] *= r;
    }
    if (flags & F_SWIGLU) {
#pragma unroll
        for (int t = 0; t < NT; t += 2) {
            __bf16 o[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float gt = (float)(__bf16)acc[t][j], up = (float)(__bf16)acc[t + 1][j];
                o[j] = (__bf16)(gt / (1.f + __expf(-gt)) * up);
            }
            *(uint2*)(y + ((tile0 + t) >> 1) * 16 + lg * 4) = __builtin_bit_cast(uint2, o);
        }
        return;
    }
    unsigned long long best = 0;
#pragma unroll
    for (int t = 0; t < NT; t++) {
        __bf16 o[4];
        const __bf16* rb = (const __bf16*)&rv[t];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = (tile0 + t) * 16 + lg * 4 + j;
            float v = acc[t][j];
            v += (float)rb[j];
            o[j] = (__bf16)v;
            if (flags & F_ARGMAX) {
                const unsigned long long k = am_key((float)o[j], n);
                best = k > best ? k : best;
            }
        }
        *(uint2*)(y + (tile0 + t) * 16 + lg * 4) = __builtin_bit_cast(uint2, o);
    }
    if (flags & F_ARGMAX) {     // lanes 0, 16, 32, 48
#pragma unroll
        for (int o = 16; o < 64; o <<= 1) {
            const unsigned long long k = (unsigned long long)__shfl_xor((long long)best, o);
            best = k > best ? k : best;
        }
        if (lane == 0) __hip_atomic_fetch_max(am + blockIdx.x % AM_S, best, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
}

#define GVI(NT, W, D, CW, RMS, I8) \
    if (nt == NT && waves == W && d == D && KB == sk * W * CW && rms == RMS && i8 == I8) { \
        hipLaunchKernelGGL((w4_gemv_kernel<NT, W, D, CW, RMS, I8>), dim3(N / 16 / NT, sk), dim3(W * 64), \
            W * CW * ((I8 ? 128 : 256) + NT * 48 + 8), stream, (const __bf16*)x, (const u32x4*)qw, (const u32x4*)meta, \
            (__bf16*)y, (const __bf16*)res, N, K, 0x000F000Fu, 0x43004300u, flags, eps, (unsigned long long*)am, ws, cnt); \
        return 0; }
#define GVR(NT, W, D, CW) GVI(NT, W, D, CW, false, false) GVI(NT, W, D, CW, true, false) \
                          GVI(NT, W, D, CW, false, true) GVI(NT, W, D, CW, true, true)
#define GVD(NT, W, CW) GVR(NT, W, 1, CW) GVR(NT, W, 2, CW)
#define GVC(W, CW) GVD(1, W, CW) GVD(2, W, CW)
#define GVRN(NT, W, D, CW) GVI(NT, W, D, CW, false, false) GVI(NT, W, D, CW, true, false)
#define GVN(W, CW) GVRN(1, W, 1, CW) GVRN(1, W, 2, CW) GVRN(2, W, 1, CW) GVRN(2, W, 2, CW)
#define GVDD(W, CW) GVRN(1, W, 3, CW) GVRN(1, W, 4, CW) GVRN(2, W, 3, CW) GVRN(2, W, 4, CW)

extern "C" int w4_gemv(const void* x, const void* qw, const void* meta, void* y, const void* res, int N, int K,
                       int nt, int waves, int d, int sk, int flags, int rms, int i8, float eps, void* am, float* ws,
                       int* cnt, hipStream_t stream) {
    const int KB = K / 128;
    if (KB % (waves * sk)) return -2;
    const int cw = KB / waves / sk;
    if (waves * cw * ((i8 ? 128 : 256) + nt * 48 + 8) + (waves - 1) * nt * 1024 + 4 * waves > 65536) return -3;
    // (waves, k blocks per wave) for K = 5120, 17408, 6144, 10240
    GVC(4, 10) GVC(8, 5) GVC(4, 34) GVC(8, 17) GVC(4, 12) GVC(8, 6) GVC(16, 3) GVC(4, 20) GVC(8, 10) GVC(16, 5)
    // split k (KB = sk W CW)
    GVN(4, 5) GVN(2, 10) GVN(4, 17) GVN(2, 17) GVN(4, 6) GVN(8, 3) GVN(4, 3) GVN(2, 12) GVN(2, 20) GVN(4, 4)
    // deeper rings (more weight bytes in flight per wave)
    GVDD(4, 10) GVDD(8, 5) GVDD(8, 17) GVDD(4, 17) GVDD(8, 6) GVDD(4, 12) GVDD(8, 10) GVDD(4, 20)
    return -1;
}

#define INST(NT, G, W, RMS) \
    if (nt == NT && g == G && waves == W && rms == RMS) { \
        hipLaunchKernelGGL((w4_gemm_kernel<NT, G, W, true, RMS>), dim3(N / 16 / NT, ksplit), dim3(W * 64), \
            NT * ((K / 128 + ksplit - 1) / ksplit) * (128 / G) * 48, stream, \
            (const __bf16*)x, (const u32x4*)qw, (const u32x4*)meta, (__bf16*)y, \
            (const __bf16*)res, ws, cnt, M, N, K, 0x000F000Fu, 0x43004300u, flags, eps, \
            (unsigned long long*)am); \
        return 0; }

#define INST_W(NT, G, RMS) INST(NT, G, 1, RMS) INST(NT, G, 2, RMS) INST(NT, G, 4, RMS) INST(NT, G, 8, RMS)
#define INST_G(G, RMS) INST_W(1, G, RMS) INST_W(2, G, RMS) INST_W(4, G, RMS) INST_W(8, G, RMS) \
                       INST(1, G, 16, RMS) INST(2, G, 16, RMS)

#define GVM(NT, W, D, CW, RMS, MODE) \
    if (nt == NT && waves == W && d == D && KB == W * CW && rms == RMS && mode == MODE) { \
        hipLaunchKernelGGL((w4_gemv_kernel<NT, W, D, CW, RMS, false, MODE>), dim3(N / 16 / NT), dim3(W * 64), \
            W * CW * (256 + NT * 48 + 8), stream, (const __bf16*)x, (const u32x4*)qw, (const u32x4*)meta, \
            (__bf16*)y, (const __bf16*)res, N, K, 0x000F000Fu, 0x43004300u, flags, eps, (unsigned long long*)am, \
            nullptr, nullptr); \
        return 0; }
#define GVM2(NT, W, D, CW, RMS) GVM(NT, W, D, CW, RMS, 1) GVM(NT, W, D, CW, RMS, 2)
extern "C" int w4_gemv_mode(const void* x, const void* qw, const void* meta, void* y, const void* res, int N, int K,
                            int nt, int waves, int d, int flags, int rms, int mode, float eps, void* am, hipStream_t stream) {
    const int KB = K / 128;
    GVM2(2, 4, 2, 10, true) GVM2(2, 8, 1, 5, true) GVM2(2, 8, 2, 5, true) GVM2(2, 8, 2, 17, false) GVM2(2, 4, 2, 34, false)
    GVM2(2, 4, 1, 10, false) GVM2(2, 4, 2, 10, false)
    return -1;
}

extern "C" int w4_gemm(const void* x, const void* qw, const void* meta, void* y, const void* res,
                       float* ws, int* cnt, int M, int N, int K, int g, int nt, int waves, int ksplit,
                       int rms, int flags, float eps, void* am, hipStream_t stream) {
    INST_G(128, 0) INST_G(64, 0) INST_G(128, 1) INST_G(64, 1)
    return -1;
}

// memory read microbenchmark: every thread reads 16-byte words, grid-stride
template <int U>
__global__ __launch_bounds__(256) void rd_kernel(const u32x4* __restrict__ p, size_t n, uint32_t* out) {
    uint32_t a = 0;
    size_t i = (size_t)blockIdx.x * 256 * U + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * 256 * U;
    for (; i < n; i += stride) {
        u32x4 v[U];
#pragma unroll
        for (int u = 0; u < U; u++) v[u] = (i + u * 256 < n) ? __builtin_nontemporal_load(p + i + u * 256) : u32x4{0u,0u,0u,0u};
#pragma unroll
        for (int u = 0; u < U; u++) a ^= v[u][0] ^ v[u][1] ^ v[u][2] ^ v[u][3];
    }
    if (a == 0x12345678u) out[0] = a;
}

extern "C" int rd_bench(const void* p, size_t nbytes, void* out, int blocks, int u, hipStream_t stream) {
    size_t n = nbytes / 16;
    if (u == 1) hipLaunchKernelGGL(rd_kernel<1>, dim3(blocks), dim3(256), 0, stream, (const u32x4*)p, n, (uint32_t*)out);
    else if (u == 4) hipLaunchKernelGGL(rd_kernel<4>, dim3(blocks), dim3(256), 0, stream, (const u32x4*)p, n, (uint32_t*)out);
    else return -1;
    return 0;
}
"""
