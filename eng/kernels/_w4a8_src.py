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
#ifndef DEPTH
#define DEPTH 2      // k blocks in flight per wave: 2, 3 or 4
#endif

enum { F_SWIGLU = 1, F_ACC = 2 };

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

// Quantize one group of 128 of one row: 16 lanes (sub = lane % 16) x 8 values; x points at the row.
// nw (may be null): RMSNorm weight; x (1 + nw) is quantized, sum of x^2 is of the raw x (the GEMM applies 1/rms).
// All 16 lanes must call it (shuffles); act false: lane contributes nothing and writes nothing.
// raw: this lane's 8 values of the group
__device__ __forceinline__ void quant_group(u32x4 raw, const __bf16* nw, int g, int m, int K, int sub, bool act,
                                            int8_t* xq, float2* xg, __bf16* xzp) {
    const int NG = K >> 7, NGp = (NG + 31) & ~31;
    float v[8];
    float amax = 0.f, sq = 0.f;
    if (act) {
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
    if (act) {
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
    if (act && sub == 0) {
        xg[(size_t)m * NG + g] = make_float2(sx / 16.f, sq);
        xzp[(size_t)m * NGp + g] = (__bf16)(sx * (float)s);
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
    // split K: workgroup (x, s) takes groups [g0, g1); its waves take them round robin
    const int S = gridDim.y, sk = blockIdx.y;
    const int g0 = sk * NG / S, g1 = (sk + 1) * NG / S;
    constexpr int KS = WAVES;
    const int kbeg = g0 + wave, kend = g1;

    const __amdgpu_buffer_rsrc_t xr = __builtin_amdgcn_make_buffer_rsrc((void*)xq, 0, M * K, 0x00020000);
    const __amdgpu_buffer_rsrc_t gr = __builtin_amdgcn_make_buffer_rsrc((void*)xg, 0, M * NG * 8, 0x00020000);
    const __amdgpu_buffer_rsrc_t zr = __builtin_amdgcn_make_buffer_rsrc((void*)xzp, 0, M * NGp * 2, 0x00020000);
    const int xoff = m * K + lg * 16;
    const u32x4* qbase = qw + (size_t)tile0 * NG * 64 + lane;

    auto load = [&](Frag<NT>& f, int kb) {
        kb = min(kb, kend - 1);
#ifdef X_NOLOAD   // experiment: every load hits the same (cached) block
        kb = 0;
#endif
#pragma unroll
        for (int t = 0; t < NT; t++) f.w[t] = __builtin_nontemporal_load(qbase + ((size_t)t * NG + kb) * 64);
#ifdef X_NOX   // experiment: no x loads
        f.xl = i32x4{kb, 1, 2, 3}; f.xh = f.xl; f.g = make_float2(1.f, 1.f);
#else
        f.xl = __builtin_bit_cast(i32x4, __builtin_amdgcn_raw_buffer_load_b128(xr, xoff + kb * 128, 0, 0));
        f.xh = __builtin_bit_cast(i32x4, __builtin_amdgcn_raw_buffer_load_b128(xr, xoff + kb * 128 + 64, 0, 0));
        f.g = __builtin_bit_cast(float2, __builtin_amdgcn_raw_buffer_load_b64(gr, (m * NG + kb) * 8, 0, 0));
#endif
    };
    // the scales of the NT tiles (groups [g0, g1), global layout [t][NG][2 x 16 bytes] kept in LDS) and the
    // zero-point operands of this wave's first chunk of 32 groups are requested first, then the first DEPTH k blocks
    extern __shared__ u32x4 lsc[];
    const int slen = (g1 - g0) * 2, n16 = NT * slen;
    const u32x4* ssrc = msc + (size_t)tile0 * NG * 2;
    auto sidx = [&](int i) { const int t = i / slen; return t * NG * 2 + g0 * 2 + i - t * slen; };
    constexpr int SCB = 4;
    u32x4 sv[SCB];
#ifndef X_NOPRO
#pragma unroll
    for (int u = 0; u < SCB; u++) {
        const int i = threadIdx.x + u * WAVES * 64;
        if (i < n16) sv[u] = ssrc[sidx(i)];
    }
#endif
    const bool zt = sk == 0;     // the zero-point term is added by split 0
    const int c0 = wave;
    bf16x8 zb, za[NT];
#ifndef X_NOPRO
    if (zt && c0 * 32 < NG) {
        zb = __builtin_bit_cast(bf16x8, __builtin_amdgcn_raw_buffer_load_b128(zr, (m * NGp + c0 * 32 + lg * 8) * 2, 0, 0));
#pragma unroll
        for (int t = 0; t < NT; t++)
            za[t] = __builtin_bit_cast(bf16x8, mzp[((size_t)(tile0 + t) * 16 + m) * (NGp / 8) + c0 * 4 + lg]);
    }
#endif
    Frag<NT> f0, f1, f2, f3;
    load(f0, kbeg);
    load(f1, kbeg + KS);
    if (DEPTH > 2) load(f2, kbeg + 2 * KS);
    if (DEPTH > 3) load(f3, kbeg + 3 * KS);
    BAR

    f32x4 acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = f32x4{0.f, 0.f, 0.f, 0.f};
    float ss = 0.f;
#ifndef X_NOPRO   // experiment: no scale copy and no zero-point term
#pragma unroll
    for (int u = 0; u < SCB; u++) {
        const int i = threadIdx.x + u * WAVES * 64;
        if (i < n16) lsc[sidx(i)] = sv[u];
    }
    for (int i = threadIdx.x + SCB * WAVES * 64; i < n16; i += WAVES * 64) lsc[sidx(i)] = ssrc[sidx(i)];
    if (zt && c0 * 32 < NG) {
#pragma unroll
        for (int t = 0; t < NT; t++) acc[t] = MFMA_BF(za[t], zb, acc[t], 0, 0, 0);
    }
    // zero-point term: acc = sum_g (-sw (z - 8)) (sx S), chunks of 32 groups spread over the waves
    for (int c = wave + WAVES; zt && c * 32 < NG; c += WAVES) {
        const bf16x8 b = __builtin_bit_cast(bf16x8, __builtin_amdgcn_raw_buffer_load_b128(zr, (m * NGp + c * 32 + lg * 8) * 2, 0, 0));
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const bf16x8 a = __builtin_bit_cast(bf16x8, mzp[((size_t)(tile0 + t) * 16 + m) * (NGp / 8) + c * 4 + lg]);
            acc[t] = MFMA_BF(a, b, acc[t], 0, 0, 0);
        }
    }
#endif

    const char* ls = (const char*)lsc;
    auto mma = [&](const Frag<NT>& f, Pend<NT>& p) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
#ifdef X_NOMATH   // experiment: loads only
            p.a[t] = __builtin_bit_cast(i32x4, f.w[t]) ^ f.xl ^ f.xh;
            continue;
#endif
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
#ifdef X_NOMATH
            acc[t][0] += (float)(p.a[t][0] ^ p.a[t][1] ^ p.a[t][2] ^ p.a[t][3]);
            continue;
#endif
            const uint2 sw = *(const uint2*)(ls + ((t * NG + kb) * 16 + lg * 4) * 2);
            const __half2 s01 = __builtin_bit_cast(__half2, sw.x);
            const __half2 s23 = __builtin_bit_cast(__half2, sw.y);
            const float sj[4] = {__low2float(s01), __high2float(s01), __low2float(s23), __high2float(s23)};
#pragma unroll
            for (int j = 0; j < 4; j++) acc[t][j] += (sj[j] * sx) * (float)p.a[t][j];
        }
    };

    // ring of DEPTH k blocks in flight per wave; the scale epilogue of a block runs after the MFMAs of the next
    {
        Pend<NT> p0, p1, p2, p3;
        Pend<NT>& plast = DEPTH == 2 ? p1 : DEPTH == 3 ? p2 : p3;
#pragma unroll
        for (int t = 0; t < NT; t++) plast.a[t] = i32x4{0, 0, 0, 0};
        plast.g = make_float2(0.f, 0.f);
        int kb = kbeg;
        __syncthreads();
#define STEP(F, PC, PP) \
        BAR \
        mma(F, PC); \
        load(F, kb + DEPTH * KS); \
        epi(PP, max(kb - KS, kbeg)); \
        BAR \
        kb += KS; \
        if (kb >= kend) { epi(PC, kb - KS); break; }
        if (kb < kend) {
            while (true) {
                if (DEPTH == 2) { STEP(f0, p0, p1) STEP(f1, p1, p0) }
                if (DEPTH == 3) { STEP(f0, p0, p2) STEP(f1, p1, p0) STEP(f2, p2, p1) }
                if (DEPTH == 4) { STEP(f0, p0, p3) STEP(f1, p1, p0) STEP(f2, p2, p1) STEP(f3, p3, p2) }
            }
        }
#undef STEP
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
    const bool mok = m < M;
    if (RMS) {
        // every lane group holds the same partial sums for row m (each lane loads its row's xg)
        const float r = rsqrtf(ss / (float)K + eps);
#pragma unroll
        for (int t = 0; t < NT; t++) acc[t] *= r;
    }
    const bool sw = flags & F_SWIGLU;
    const int Ny = sw ? N >> 1 : N;
    auto store4 = [&](size_t o, const float* v) {     // 4 consecutive outputs, 8 bytes
        const uint2 u = make_uint2(__builtin_bit_cast(unsigned short, (__bf16)v[0]) | ((uint32_t)__builtin_bit_cast(unsigned short, (__bf16)v[1]) << 16),
                                   __builtin_bit_cast(unsigned short, (__bf16)v[2]) | ((uint32_t)__builtin_bit_cast(unsigned short, (__bf16)v[3]) << 16));
        *(uint2*)(y + o) = u;
    };
    if (mok && (flags & F_ACC)) {     // fp32 y += partial sum (split K); quant_x adds y to the residual
        float* ya = (float*)y;
#pragma unroll
        for (int t = 0; t < NT; t++)
#pragma unroll
            for (int j = 0; j < 4; j++) unsafeAtomicAdd(ya + (size_t)m * N + (tile0 + t) * 16 + lg * 4 + j, acc[t][j]);
    } else if (mok) {
        if (sw) {
#pragma unroll
            for (int t = 0; t < NT; t += 2) {
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float gt = (float)(__bf16)acc[t][j], up = (float)(__bf16)acc[t + 1][j];
                    v[j] = gt / (1.f + __expf(-gt)) * up;
                }
                store4((size_t)m * Ny + ((tile0 + t) >> 1) * 16 + lg * 4, v);
            }
        } else {
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const size_t o = (size_t)m * N + (tile0 + t) * 16 + lg * 4;
                float v[4] = {acc[t][0], acc[t][1], acc[t][2], acc[t][3]};
                if (res) {
                    const uint2 r = *(const uint2*)(res + o);
                    v[0] += __uint_as_float(r.x << 16);
                    v[1] += __uint_as_float(r.x & 0xFFFF0000u);
                    v[2] += __uint_as_float(r.y << 16);
                    v[3] += __uint_as_float(r.y & 0xFFFF0000u);
                }
                store4(o, v);
            }
        }
    }
}

// x (M, K) bf16 -> xq, xg, xzp. Block of 64 threads = 4 groups. grid (NGp / 4, M). Groups in [NG, NGp): xzp = 0.
// acc (may be null): fp32 (M, K) sums of F_ACC GEMMs; x += acc (in place, rounded to bf16) and acc = 0 first.
__global__ __launch_bounds__(64) void quant_x_kernel(__bf16* x, const __bf16* nw, int8_t* xq, float2* xg,
                                                     __bf16* xzp, float* acc, int K) {
    const int NG = K >> 7, NGp = (NG + 31) & ~31;
    const int m = blockIdx.y;
    const int g = blockIdx.x * 4 + (threadIdx.x >> 4);
    const int sub = threadIdx.x & 15;
    const size_t o = (size_t)m * K + g * 128 + sub * 8;
    u32x4 raw = g < NG ? *(const u32x4*)(x + o) : u32x4{0u, 0u, 0u, 0u};
    if (acc && g < NG) {
        float4* ap = (float4*)(acc + o);
        const float4 a0 = ap[0], a1 = ap[1];
        const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float lo = __uint_as_float(raw[i] << 16) + a[2 * i];
            const float hi = __uint_as_float(raw[i] & 0xFFFF0000u) + a[2 * i + 1];
            raw[i] = __builtin_bit_cast(unsigned short, (__bf16)lo) | ((uint32_t)__builtin_bit_cast(unsigned short, (__bf16)hi) << 16);
        }
        *(u32x4*)(x + o) = raw;
        ap[0] = make_float4(0.f, 0.f, 0.f, 0.f);
        ap[1] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    quant_group(raw, nw, g, m, K, sub, g < NG, xq, xg, xzp);
    if (sub == 0 && g >= NG && g < NGp) xzp[(size_t)m * NGp + g] = (__bf16)0.f;
}

#define INST(NT, W, RMS) \
    if (nt == NT && waves == W && rms == RMS) { \
        hipLaunchKernelGGL((w4a8_kernel<NT, W, RMS>), dim3(N / 16 / NT, splits), dim3(W * 64), NT * (K / 128) * 32, stream, \
            (const int8_t*)xq, (const float2*)xg, (const __bf16*)xzp, (const u32x4*)qw, (const u32x4*)msc, \
            (const u32x4*)mzp, (__bf16*)y, (const __bf16*)res, M, N, K, flags, eps); \
        return 0; }
#define INST_W(NT, RMS) INST(NT, 1, RMS) INST(NT, 2, RMS) INST(NT, 4, RMS) INST(NT, 8, RMS)
#define INST_R(RMS) INST_W(1, RMS) INST_W(2, RMS) INST_W(4, RMS) INST_W(8, RMS)

extern "C" int w4a8_gemm(const void* xq, const void* xg, const void* xzp, const void* qw, const void* msc,
                         const void* mzp, void* y, const void* res, int M, int N, int K, int nt, int waves,
                         int rms, int flags, float eps, int splits, hipStream_t stream) {
    INST_R(0) INST_R(1)
    return -1;
}

extern "C" int w4a8_quant_x(void* x, const void* nw, void* xq, void* xg, void* xzp, void* acc, int M, int K,
                            hipStream_t stream) {
    const int NGp = ((K >> 7) + 31) & ~31;
    hipLaunchKernelGGL(quant_x_kernel, dim3(NGp / 4, M), dim3(64), 0, stream, (__bf16*)x, (const __bf16*)nw, (int8_t*)xq,
                       (float2*)xg, (__bf16*)xzp, (float*)acc, K);
    return 0;
}
"""
