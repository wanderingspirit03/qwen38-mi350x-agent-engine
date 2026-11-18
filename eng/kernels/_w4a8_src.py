HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

// W4A8 decode GEMM. Layouts (see w4a8.py):
//   qw   int32 [N/16][NG][64 lanes][4]: lane l = row 16t + l%16, lane group lg = l/16, dword d, byte b holds
//        v(k = 128g + 16lg + 4d + b) in its low nibble and v(k + 64) in its high nibble, v = u ^ 8 (u - 8 as int4).
//        lo operand (q << 4) & 0xF0F0F0F0 and hi operand q & 0xF0F0F0F0 are int8 16 (u - 8).
//   msc  fp16 [N/16][NG][16]   weight scales s_w
//   mzp  bf16 [N/16][16][NGp]  -s_w (z - 8), NGp = NG rounded up to 32 (zero padded)
//   xq   int8 [M][K]           x = sx q per (row, group of 128)
//   xg   float2 [M][NG]        (sx / 16, sum of x^2 over the group)
//   xzp  bf16 [M][NGp]         sx * sum(q) over the group (zero padded)
// y[n][m] = sum_g sw sx/16 (acc_lo16 + acc_hi16) - sum_g sw (z - 8) sx S.
// MFMA 16x16: D[n][m], lane l holds D[4(l/16) + j][l%16].

#define MFMA_I8 __builtin_amdgcn_mfma_i32_16x16x64_i8
#define MFMA_BF __builtin_amdgcn_mfma_f32_16x16x32_bf16
#define BAR asm volatile("" ::: "memory"); __builtin_amdgcn_sched_barrier(0);

enum { F_SWIGLU = 1 };

template <int NT>
struct Frag {
    u32x4 w[NT];
    i32x4 xl, xh;
    float2 g;
};

template <int NT>
struct Pend {
    i32x4 a[NT];
    float2 g;
};

__device__ __forceinline__ void deq(u32x4 q, i32x4& lo, i32x4& hi) {
#pragma unroll
    for (int d = 0; d < 4; d++) {
        lo[d] = (int)((q[d] << 4) & 0xF0F0F0F0u);
        hi[d] = (int)(q[d] & 0xF0F0F0F0u);
    }
}

template <int NT, int WAVES, bool RMS>
__global__ __launch_bounds__(WAVES * 64)
void w4a8_kernel(const int8_t* xq, const float2* xg, const __bf16* xzp, const u32x4* qw, const u32x4* msc,
                 const u32x4* mzp, __bf16* y, const __bf16* res, int M, int N, int K, int flags, float eps) {
    const int lane = threadIdx.x & 63;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int lg = lane >> 4;
    const int m = lane & 15;
    const int NG = K >> 7;
    const int NGp = (NG + 31) & ~31;
    const int tile0 = blockIdx.x * NT;

    const __amdgpu_buffer_rsrc_t xr = __builtin_amdgcn_make_buffer_rsrc((void*)xq, 0, M * K, 0x00020000);
    const __amdgpu_buffer_rsrc_t gr = __builtin_amdgcn_make_buffer_rsrc((void*)xg, 0, M * NG * 8, 0x00020000);
    const __amdgpu_buffer_rsrc_t zr = __builtin_amdgcn_make_buffer_rsrc((void*)xzp, 0, M * NGp * 2, 0x00020000);
    const int xoff = m * K + lg * 16;
    const u32x4* qbase = qw + (size_t)tile0 * NG * 64 + lane;

    // scales of the NT tiles -> LDS (NT * NG * 32 bytes, contiguous in global memory)
    extern __shared__ u32x4 lsc[];
    {
        const int n16 = NT * NG * 2;
        const u32x4* src = msc + (size_t)tile0 * NG * 2;
        for (int i = threadIdx.x; i < n16; i += WAVES * 64) lsc[i] = src[i];
    }

    f32x4 acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = f32x4{0.f, 0.f, 0.f, 0.f};
    float ss = 0.f;

    // zero-point term: acc = sum_g (-sw (z - 8)) (sx S), chunks of 32 groups spread over the waves
    for (int c = wave; c * 32 < NG; c += WAVES) {
        const bf16x8 b = __builtin_bit_cast(bf16x8, __builtin_amdgcn_raw_buffer_load_b128(zr, (m * NGp + c * 32 + lg * 8) * 2, 0, 0));
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const bf16x8 a = __builtin_bit_cast(bf16x8, mzp[((size_t)(tile0 + t) * 16 + m) * (NGp / 8) + c * 4 + lg]);
            acc[t] = MFMA_BF(a, b, acc[t], 0, 0, 0);
        }
    }

    auto load = [&](Frag<NT>& f, int kb) {
        kb = min(kb, NG - 1);
#pragma unroll
        for (int t = 0; t < NT; t++) f.w[t] = __builtin_nontemporal_load(qbase + ((size_t)t * NG + kb) * 64);
        f.xl = __builtin_bit_cast(i32x4, __builtin_amdgcn_raw_buffer_load_b128(xr, xoff + kb * 128, 0, 0));
        f.xh = __builtin_bit_cast(i32x4, __builtin_amdgcn_raw_buffer_load_b128(xr, xoff + kb * 128 + 64, 0, 0));
        f.g = __builtin_bit_cast(float2, __builtin_amdgcn_raw_buffer_load_b64(gr, (m * NG + kb) * 8, 0, 0));
    };
    const char* ls = (const char*)lsc;
    auto mma = [&](const Frag<NT>& f, Pend<NT>& p) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
            i32x4 lo, hi;
            deq(f.w[t], lo, hi);
            i32x4 a = MFMA_I8(lo, f.xl, (i32x4){0, 0, 0, 0}, 0, 0, 0);
            p.a[t] = MFMA_I8(hi, f.xh, a, 0, 0, 0);
        }
        p.g = f.g;
    };
    auto epi = [&](const Pend<NT>& p, int kb) {
        const float sx = p.g.x;
        if (RMS) ss += p.g.y;
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const uint2 sw = *(const uint2*)(ls + ((t * NG + kb) * 16 + lg * 4) * 2);
            const __half2 s01 = __builtin_bit_cast(__half2, sw.x);
            const __half2 s23 = __builtin_bit_cast(__half2, sw.y);
            const float sj[4] = {__low2float(s01), __high2float(s01), __low2float(s23), __high2float(s23)};
#pragma unroll
            for (int j = 0; j < 4; j++) acc[t][j] += (sj[j] * sx) * (float)p.a[t][j];
        }
    };

    {
        Frag<NT> fa, fb;
        Pend<NT> pa, pb;
        int kb = wave;
        if (kb < NG) {
            load(fa, kb);
            load(fb, kb + WAVES);
        }
        __syncthreads();
        if (kb < NG) {
            BAR
            mma(fa, pa);
            BAR
            while (true) {
                kb += WAVES;
                if (kb >= NG) { epi(pa, kb - WAVES); break; }
                load(fa, kb + WAVES);
                BAR
                mma(fb, pb);
                epi(pa, kb - WAVES);
                BAR
                kb += WAVES;
                if (kb >= NG) { epi(pb, kb - WAVES); break; }
                load(fb, kb + WAVES);
                BAR
                mma(fa, pa);
                epi(pb, kb - WAVES);
                BAR
            }
        }
    }

    if (WAVES > 1) {
        __shared__ f32x4 red[WAVES > 1 ? WAVES - 1 : 1][NT][64];
        __shared__ float rss[WAVES > 1 ? WAVES - 1 : 1][64];
        if (wave > 0) {
#pragma unroll
            for (int t = 0; t < NT; t++) red[wave - 1][t][lane] = acc[t];
            if (RMS) rss[wave - 1][lane] = ss;
        }
        __syncthreads();
        if (wave != 0) return;
#pragma unroll
        for (int t = 0; t < NT; t++)
            for (int w = 0; w < WAVES - 1; w++) acc[t] += red[w][t][lane];
        if (RMS)
            for (int w = 0; w < WAVES - 1; w++) ss += rss[w][lane];
    }
    if (m >= M) return;
    if (RMS) {
        // every lane group holds the same partial sums for row m (each lane loads its row's xg)
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
                const float gt = (float)(__bf16)acc[t][j], up = (float)(__bf16)acc[t + 1][j];
                y[(size_t)m * N2 + n] = (__bf16)(gt / (1.f + __expf(-gt)) * up);
            }
        }
        return;
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const size_t o = (size_t)m * N + (tile0 + t) * 16 + lg * 4 + j;
            float v = acc[t][j];
            if (res) v += (float)res[o];
            y[o] = (__bf16)v;
        }
    }
}

// x (M, K) bf16 -> xq, xg, xzp. Block of 64 threads = 4 groups, 16 lanes x 8 values per group.
// grid (NGp / 4, M). Groups g in [NG, NGp) only write xzp = 0.
// nw (may be null): RMSNorm weight; x (1 + nw) is quantized, sum of x^2 is of the raw x (the GEMM applies 1/rms).
__global__ __launch_bounds__(64) void quant_x_kernel(const __bf16* x, const __bf16* nw, int8_t* xq, float2* xg,
                                                     __bf16* xzp, int K) {
    const int NG = K >> 7, NGp = (NG + 31) & ~31;
    const int m = blockIdx.y;
    const int g = blockIdx.x * 4 + (threadIdx.x >> 4);
    const int sub = threadIdx.x & 15;
    float v[8];
    float amax = 0.f, sq = 0.f;
    if (g < NG) {
        const u32x4 raw = *(const u32x4*)(x + (size_t)m * K + g * 128 + sub * 8);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            v[2 * i] = __uint_as_float(raw[i] << 16);
            v[2 * i + 1] = __uint_as_float(raw[i] & 0xFFFF0000u);
        }
#pragma unroll
        for (int i = 0; i < 8; i++) sq += v[i] * v[i];
        if (nw) {
            const u32x4 wr = *(const u32x4*)(nw + g * 128 + sub * 8);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                v[2 * i] *= 1.f + __uint_as_float(wr[i] << 16);
                v[2 * i + 1] *= 1.f + __uint_as_float(wr[i] & 0xFFFF0000u);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; i++) amax = fmaxf(amax, fabsf(v[i]));
    }
#pragma unroll
    for (int o = 8; o; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, o, 16));
        sq += __shfl_xor(sq, o, 16);
    }
    const float sx = amax / 127.f;
    const float inv = amax > 0.f ? 127.f / amax : 0.f;
    int s = 0;
    if (g < NG) {
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int q = (int)rintf(v[i] * inv);
            s += q;
            if (i < 4) w0 |= (uint32_t)(q & 0xFF) << (8 * i);
            else w1 |= (uint32_t)(q & 0xFF) << (8 * (i - 4));
        }
        *(uint2*)(xq + (size_t)m * K + g * 128 + sub * 8) = make_uint2(w0, w1);
    }
#pragma unroll
    for (int o = 8; o; o >>= 1) s += __shfl_xor(s, o, 16);
    if (sub == 0 && g < NGp) {
        if (g < NG) xg[(size_t)m * NG + g] = make_float2(sx / 16.f, sq);
        xzp[(size_t)m * NGp + g] = (__bf16)(sx * (float)s);
    }
}

#define INST(NT, W, RMS) \
    if (nt == NT && waves == W && rms == RMS) { \
        hipLaunchKernelGGL((w4a8_kernel<NT, W, RMS>), dim3(N / 16 / NT), dim3(W * 64), NT * (K / 128) * 32, stream, \
            (const int8_t*)xq, (const float2*)xg, (const __bf16*)xzp, (const u32x4*)qw, (const u32x4*)msc, \
            (const u32x4*)mzp, (__bf16*)y, (const __bf16*)res, M, N, K, flags, eps); \
        return 0; }
#define INST_W(NT, RMS) INST(NT, 1, RMS) INST(NT, 2, RMS) INST(NT, 4, RMS) INST(NT, 8, RMS)
#define INST_R(RMS) INST_W(1, RMS) INST_W(2, RMS) INST_W(4, RMS) INST_W(8, RMS)

extern "C" int w4a8_gemm(const void* xq, const void* xg, const void* xzp, const void* qw, const void* msc,
                         const void* mzp, void* y, const void* res, int M, int N, int K, int nt, int waves,
                         int rms, int flags, float eps, hipStream_t stream) {
    INST_R(0) INST_R(1)
    return -1;
}

extern "C" int w4a8_quant_x(const void* x, const void* nw, void* xq, void* xg, void* xzp, int M, int K,
                            hipStream_t stream) {
    const int NGp = ((K >> 7) + 31) & ~31;
    hipLaunchKernelGGL(quant_x_kernel, dim3(NGp / 4, M), dim3(64), 0, stream, (const __bf16*)x, (const __bf16*)nw, (int8_t*)xq,
                       (float2*)xg, (__bf16*)xzp, K);
    return 0;
}
"""
