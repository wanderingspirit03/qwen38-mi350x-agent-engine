HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

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

enum { F_SWIGLU = 1, F_XDUP = 2 };   // F_XDUP: x has K/2 columns, k block kb reads x block kb/2

// After the k loop: split-k reduction (the last of the ksplit workgroups of a unit adds the partials up),
// 1/rms scaling, then SwiGLU or the (optional) residual add, and the bf16 store. unit: index of the
// independent group of NT row tiles (counter and sum(x^2) slot). acc[NT] holds x.x^T if RMS.
template <int NT, bool RMS>
__device__ __forceinline__ void finish(f32x4* acc, int unit, int tile0, int lane, __bf16* y, const __bf16* res,
                                       float* ws, int* cnt, int M, int N, int K, int flags, float eps) {
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
#pragma unroll
    for (int t = 0; t < NT; t++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = (tile0 + t) * 16 + lg * 4 + j;
            const size_t o = (size_t)m * N + n;
            float v = acc[t][j];
            if (res) v += (float)res[o];
            y[o] = (__bf16)v;
        }
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
                    int M, int N, int K, uint32_t mask, uint32_t magic, int flags, float eps) {
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
    finish<NT, RMS>(acc, blockIdx.x, tile0, lane, y, res, ws, cnt, M, N, K, flags, eps);
}

#define INST(NT, G, W, RMS) \
    if (nt == NT && g == G && waves == W && rms == RMS) { \
        hipLaunchKernelGGL((w4_gemm_kernel<NT, G, W, true, RMS>), dim3(N / 16 / NT, ksplit), dim3(W * 64), \
            NT * ((K / 128 + ksplit - 1) / ksplit) * (128 / G) * 48, stream, \
            (const __bf16*)x, (const u32x4*)qw, (const u32x4*)meta, (__bf16*)y, \
            (const __bf16*)res, ws, cnt, M, N, K, 0x000F000Fu, 0x43004300u, flags, eps); \
        return 0; }

#define INST_W(NT, G, RMS) INST(NT, G, 1, RMS) INST(NT, G, 2, RMS) INST(NT, G, 4, RMS) INST(NT, G, 8, RMS)
#define INST_G(G, RMS) INST_W(1, G, RMS) INST_W(2, G, RMS) INST_W(4, G, RMS) INST_W(8, G, RMS) \
                       INST(1, G, 16, RMS) INST(2, G, 16, RMS)

extern "C" int w4_gemm(const void* x, const void* qw, const void* meta, void* y, const void* res,
                       float* ws, int* cnt, int M, int N, int K, int g, int nt, int waves, int ksplit,
                       int rms, int flags, float eps, hipStream_t stream) {
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
