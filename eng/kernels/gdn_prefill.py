"""Chunked gated delta rule prefill for one sequence: Triton (per-chunk parallel parts) + HIP (the scan).

Entry (after eng/csrc/gdn.hip gdn_conv):
  gdn_prefill(q, k, v, g, beta, state, state_kv=False, out=None, z=None, gw=None) -> o
    q, k: (T, 16, 128) bf16 raw (l2norm and the 128**-0.5 scale of q are applied inside); v: (T, 48, 128) bf16;
    g, beta: (T, 48) fp32; row strides free. Value head h uses key head h // 3.
    state: (48, 128, 128) fp32 [h, v, k] (any strides; [h, k, v] with state_kv=True), updated in place.
    Returns o (T, 48, 128) bf16, or with z (T, 6144, row stride free) and gw (128,) given, the gated RMSNorm
    rmsnorm_gated(o, z, gw) as (T, 6144) bf16 (what eng.hip.gated_norm computes).

Semantics: eng.ref_ops.gdn_recurrent. Three kernels over chunks of 64 positions:
  1. _wy_kernel (Triton, grid chunks x 48): l2norm, G = cumsum(g) in the chunk;
     A = strictly lower beta_i k_i.k_j exp(G_i - G_j); X = (I + A)^-1 from 16x16 blocks in fp32;
     w = X (beta exp(G) k), u = X (beta v), kd = exp(G_last - G) k; stored bf16 (u and kd transposed).
  2. gdn_scan (HIP, eng/csrc/gdn_scan.hip, 2 workgroups per head of 4 waves x 16 value columns, the fp32 state
     in registers, the chunk's w and kd shared through LDS):
     per chunk h_c = S, vn = u - w S, S = exp(G_last) S + kd^T vn  (bf16 MFMA operands, fp32 accumulation).
  3. _out_kernel (Triton, grid chunks x 48): o = (exp(G) q) h_c + (q k^T exp(G_i - G_j), j <= i) vn, then
     optionally the gated RMSNorm.
"""
import ctypes
import os

import torch
import triton
import triton.language as tl

from .. import hip as _H

C = 64
D = 128
HK = 16
HV = 48
WY_WARPS = 1
SCAN_NW = 2   # waves per scan workgroup (8 / SCAN_NW workgroups per head); 2 measured fastest
WY_HIP = os.environ.get("ENG_WY_HIP", "1") == "1"   # eng/csrc/gdn_wy.hip instead of the Triton _wy_kernel

_lib = _H.lib
_lib.gdn_wy.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int] * 3 + [ctypes.c_long] * 5 + [ctypes.c_float, ctypes.c_void_p]
_lib.gdn_scan.argtypes = ([ctypes.c_int] + [ctypes.c_void_p] * 7 + [ctypes.c_long] * 3 + [ctypes.c_void_p, ctypes.c_long]
                          + [ctypes.c_int] * 5 + [ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p])


@triton.jit
def _fdot(a, b):
    return tl.dot(a, b, input_precision="ieee")


@triton.jit
def _inv16(A):
    """(I + A)^-1 for a strictly lower 16x16 fp32 A, by doubling: X_2s = X_s - X_s C_s X_s."""
    li = tl.arange(0, 16)
    i = li[:, None]
    j = li[None, :]
    X = tl.where(i == j, 1., 0.) - tl.where(i // 2 == j // 2, A, 0.)
    for lvl in tl.static_range(1, 4):
        Cs = tl.where((i >> (lvl + 1) == j >> (lvl + 1)) & (i >> lvl != j >> lvl), A, 0.)
        X = X - _fdot(_fdot(X, Cs), X)
    return X


@triton.jit
def _ld(x, r, m, col, sx, D: tl.constexpr):
    """Rows r of one 128-column block (fp32), zero outside the sequence."""
    d = col + tl.arange(0, D)
    return tl.load(x + r[:, None] * sx + d[None, :], mask=m[:, None], other=0.).to(tl.float32)


@triton.jit
def _qk_block(q, k, qn_out, r, m, hk, sq, sk, scale, STORE: tl.constexpr, D: tl.constexpr, HK: tl.constexpr):
    """Normalized k and q rows (bf16, q scaled); the h % 3 == 0 programs store q for the scan."""
    d = tl.arange(0, D)
    kf = _ld(k, r, m, hk * D, sk, D)
    kf = kf * tl.rsqrt(tl.sum(kf * kf, 1) + 1e-6)[:, None]
    qf = _ld(q, r, m, hk * D, sq, D)
    qf = (qf * (tl.rsqrt(tl.sum(qf * qf, 1) + 1e-6) * scale)[:, None]).to(tl.bfloat16)
    if STORE:
        tl.store(qn_out + r[:, None] * (HK * D) + hk * D + d[None, :], qf)
    return kf.to(tl.bfloat16), qf


@triton.jit
def _store_M(p, qa, kb, Ga, Gb, NZ: tl.constexpr, DIAG: tl.constexpr):
    """One 16x16 block of M: q_a k_b^T exp(G_a - G_b), lower triangle incl. the diagonal if DIAG, zero if not NZ."""
    if NZ:
        x = tl.dot(qa, tl.trans(kb))
        li = tl.arange(0, 16)
        if DIAG:
            x = x * tl.exp(tl.where(li[:, None] >= li[None, :], Ga[:, None] - Gb[None, :], float("-inf")))
        else:
            x = x * tl.exp(Ga[:, None] - Gb[None, :])
        tl.store(p, x.to(tl.bfloat16))
    else:
        tl.store(p, tl.zeros((16, 16), dtype=tl.bfloat16))


@triton.jit
def _block_A(ka, kb, ba, Ga, Gb, DIAG: tl.constexpr):
    A = tl.dot(ka, tl.trans(kb)) * ba[:, None]
    li = tl.arange(0, 16)
    if DIAG:
        A = A * tl.exp(tl.where(li[:, None] > li[None, :], Ga[:, None] - Gb[None, :], float("-inf")))
    else:
        A = A * tl.exp(Ga[:, None] - Gb[None, :])
    return A


@triton.jit
def _bdot(a, b):
    return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))


@triton.jit
def _store_rows(p, r, x0, x1, x2, x3, T00, T10, T11, T20, T21, T22, T30, T31, T32, T33, HV: tl.constexpr,
                D: tl.constexpr):
    """Stores the 4 row blocks of X x (lower block triangular X) at p + rows * HV * D."""
    tl.store(p + r[:, None] * (HV * D), _bdot(T00, x0).to(tl.bfloat16))
    tl.store(p + (r + 16)[:, None] * (HV * D), (_bdot(T10, x0) + _bdot(T11, x1)).to(tl.bfloat16))
    tl.store(p + (r + 32)[:, None] * (HV * D), (_bdot(T20, x0) + _bdot(T21, x1) + _bdot(T22, x2)).to(tl.bfloat16))
    tl.store(p + (r + 48)[:, None] * (HV * D), (_bdot(T30, x0) + _bdot(T31, x1) + _bdot(T32, x2)
                                                 + _bdot(T33, x3)).to(tl.bfloat16))


@triton.jit
def _store_T(p, li, x, D: tl.constexpr):
    """x (16 rows t, 128) -> p[d * 64 + t] (transposed)."""
    d = tl.arange(0, D)
    tl.store(p + d[None, :] * 64 + li[:, None], x.to(tl.bfloat16))


@triton.jit
def _wy_kernel(q, k, v, g, beta, w_out, uT_out, kdT_out, G_out, qn_out, M_out,
               T, Tp, sq, sk, sv, sg, sb, scale, D: tl.constexpr, HK: tl.constexpr, HV: tl.constexpr):
    c = tl.program_id(0)
    h = tl.program_id(1)
    hk = h // 3
    STORE_QK = h % 3 == 0
    li = tl.arange(0, 16)
    d = tl.arange(0, D)
    r0 = c * 64 + li
    r1 = r0 + 16
    r2 = r0 + 32
    r3 = r0 + 48
    m0 = r0 < T
    m1 = r1 < T
    m2 = r2 < T
    m3 = r3 < T
    k0, q0 = _qk_block(q, k, qn_out, r0, m0, hk, sq, sk, scale, STORE_QK, D, HK)
    k1, q1 = _qk_block(q, k, qn_out, r1, m1, hk, sq, sk, scale, STORE_QK, D, HK)
    k2, q2 = _qk_block(q, k, qn_out, r2, m2, hk, sq, sk, scale, STORE_QK, D, HK)
    k3, q3 = _qk_block(q, k, qn_out, r3, m3, hk, sq, sk, scale, STORE_QK, D, HK)

    g0 = tl.load(g + r0 * sg + h, mask=m0, other=0.)
    b0 = tl.load(beta + r0 * sb + h, mask=m0, other=0.)
    g1 = tl.load(g + r1 * sg + h, mask=m1, other=0.)
    b1 = tl.load(beta + r1 * sb + h, mask=m1, other=0.)
    g2 = tl.load(g + r2 * sg + h, mask=m2, other=0.)
    b2 = tl.load(beta + r2 * sb + h, mask=m2, other=0.)
    g3 = tl.load(g + r3 * sg + h, mask=m3, other=0.)
    b3 = tl.load(beta + r3 * sb + h, mask=m3, other=0.)
    s0 = tl.sum(g0, 0)
    s1 = s0 + tl.sum(g1, 0)
    s2 = s1 + tl.sum(g2, 0)
    G0 = tl.cumsum(g0, 0)
    G1 = tl.cumsum(g1, 0) + s0
    G2 = tl.cumsum(g2, 0) + s1
    G3 = tl.cumsum(g3, 0) + s2
    Gl = s2 + tl.sum(g3, 0)
    tl.store(G_out + h * Tp + r0, G0)
    tl.store(G_out + h * Tp + r1, G1)
    tl.store(G_out + h * Tp + r2, G2)
    tl.store(G_out + h * Tp + r3, G3)

    A10 = _block_A(k1, k0, b1, G1, G0, False)
    A20 = _block_A(k2, k0, b2, G2, G0, False)
    A21 = _block_A(k2, k1, b2, G2, G1, False)
    A30 = _block_A(k3, k0, b3, G3, G0, False)
    A31 = _block_A(k3, k1, b3, G3, G1, False)
    A32 = _block_A(k3, k2, b3, G3, G2, False)
    # M = q k^T exp(G_i - G_j) for j <= i (zero above), for the output in the scan
    mp = M_out + (c * HV + h) * 4096 + li[:, None] * 64 + li[None, :]
    _store_M(mp, q0, k0, G0, G0, True, True)
    _store_M(mp + 16, q0, k0, G0, G0, False, False)
    _store_M(mp + 32, q0, k0, G0, G0, False, False)
    _store_M(mp + 48, q0, k0, G0, G0, False, False)
    _store_M(mp + 16 * 64, q1, k0, G1, G0, True, False)
    _store_M(mp + 16 * 64 + 16, q1, k1, G1, G1, True, True)
    _store_M(mp + 16 * 64 + 32, q1, k1, G1, G1, False, False)
    _store_M(mp + 16 * 64 + 48, q1, k1, G1, G1, False, False)
    _store_M(mp + 32 * 64, q2, k0, G2, G0, True, False)
    _store_M(mp + 32 * 64 + 16, q2, k1, G2, G1, True, False)
    _store_M(mp + 32 * 64 + 32, q2, k2, G2, G2, True, True)
    _store_M(mp + 32 * 64 + 48, q2, k2, G2, G2, False, False)
    _store_M(mp + 48 * 64, q3, k0, G3, G0, True, False)
    _store_M(mp + 48 * 64 + 16, q3, k1, G3, G1, True, False)
    _store_M(mp + 48 * 64 + 32, q3, k2, G3, G2, True, False)
    _store_M(mp + 48 * 64 + 48, q3, k3, G3, G3, True, True)

    T00 = _inv16(_block_A(k0, k0, b0, G0, G0, True))
    T11 = _inv16(_block_A(k1, k1, b1, G1, G1, True))
    T22 = _inv16(_block_A(k2, k2, b2, G2, G2, True))
    T33 = _inv16(_block_A(k3, k3, b3, G3, G3, True))
    T10 = -_fdot(T11, _fdot(A10, T00))
    T21 = -_fdot(T22, _fdot(A21, T11))
    T32 = -_fdot(T33, _fdot(A32, T22))
    T20 = -_fdot(T22, _fdot(A20, T00) + _fdot(A21, T10))
    T31 = -_fdot(T33, _fdot(A31, T11) + _fdot(A32, T21))
    T30 = -_fdot(T33, _fdot(A30, T00) + _fdot(A31, T10) + _fdot(A32, T20))

    # kd = exp(G_last - G) k, transposed (k-dim, t)
    kp = kdT_out + (c * HV + h) * (D * 64)
    _store_T(kp, li, k0.to(tl.float32) * tl.exp(Gl - G0)[:, None], D)
    _store_T(kp, li + 16, k1.to(tl.float32) * tl.exp(Gl - G1)[:, None], D)
    _store_T(kp, li + 32, k2.to(tl.float32) * tl.exp(Gl - G2)[:, None], D)
    _store_T(kp, li + 48, k3.to(tl.float32) * tl.exp(Gl - G3)[:, None], D)
    # w = X (beta exp(G) k)
    x0 = k0.to(tl.float32) * (b0 * tl.exp(G0))[:, None]
    x1 = k1.to(tl.float32) * (b1 * tl.exp(G1))[:, None]
    x2 = k2.to(tl.float32) * (b2 * tl.exp(G2))[:, None]
    x3 = k3.to(tl.float32) * (b3 * tl.exp(G3))[:, None]
    _store_rows(w_out + h * D + d[None, :], r0, x0, x1, x2, x3,
                T00, T10, T11, T20, T21, T22, T30, T31, T32, T33, HV, D)
    # u = X (beta v), transposed (v-dim, t)
    x0 = _ld(v, r0, m0, h * D, sv, D) * b0[:, None]
    x1 = _ld(v, r1, m1, h * D, sv, D) * b1[:, None]
    x2 = _ld(v, r2, m2, h * D, sv, D) * b2[:, None]
    x3 = _ld(v, r3, m3, h * D, sv, D) * b3[:, None]
    up = uT_out + (c * HV + h) * (D * 64)
    _store_T(up, li, _bdot(T00, x0), D)
    _store_T(up, li + 16, _bdot(T10, x0) + _bdot(T11, x1), D)
    _store_T(up, li + 32, _bdot(T20, x0) + _bdot(T21, x1) + _bdot(T22, x2), D)
    _store_T(up, li + 48, _bdot(T30, x0) + _bdot(T31, x1) + _bdot(T32, x2) + _bdot(T33, x3), D)


@triton.jit
def _gated_norm_kernel(o, z, gw, out, T, sz, so, BT: tl.constexpr, D: tl.constexpr, HV: tl.constexpr):
    """out = rmsnorm_gated(o, z, gw) for BT rows of one head (as eng.hip.gated_norm)."""
    h = tl.program_id(1)
    r = tl.program_id(0) * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    m = (r < T)[:, None]
    x = tl.load(o + r[:, None] * (HV * D) + h * D + d[None, :], mask=m, other=0.).to(tl.float32)
    x = x * tl.rsqrt(tl.sum(x * x, 1) / D + 1e-6)[:, None]
    x = (x.to(tl.bfloat16) * tl.load(gw + d)[None, :]).to(tl.float32)
    zz = tl.load(z + r[:, None] * sz + h * D + d[None, :], mask=m, other=0.).to(tl.float32)
    tl.store(out + r[:, None] * so + h * D + d[None, :], (x * (zz / (1. + tl.exp(-zz)))).to(tl.bfloat16), mask=m)


def wy(q, k, v, g, beta, w, uT, kdT, G, qn, M, T, Tp, NC, hip=True):
    """The chunk-parallel part: HIP (eng/csrc/gdn_wy.hip) or the Triton _wy_kernel (same outputs)."""
    if hip:
        rc = _lib.gdn_wy(q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(), w.data_ptr(),
                         uT.data_ptr(), kdT.data_ptr(), G.data_ptr(), qn.data_ptr(), M.data_ptr(), T, Tp, NC,
                         q.stride(0), k.stride(0), v.stride(0), g.stride(0), beta.stride(0), D ** -0.5,
                         torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"gdn_wy: hip error {rc}")
    else:
        _wy_kernel[(NC, HV)](q, k, v, g, beta, w, uT, kdT, G, qn, M,
                             T, Tp, q.stride(0), k.stride(0), v.stride(0), g.stride(0), beta.stride(0), D ** -0.5,
                             D=D, HK=HK, HV=HV, num_warps=WY_WARPS)


def gdn_prefill(q, k, v, g, beta, state, state_kv=False, out=None, z=None, gw=None, ckpt=None):
    """ckpt = (every, ck): also store the state after every, 2 every, ... tokens of this call (every a multiple of
    64, only positions <= T) into ck[0], ck[1], ... (ck: (n, 48, 128, 128) fp32, inner strides as state)."""
    T = q.shape[0]
    assert q.shape == (T, HK, D) and k.shape == (T, HK, D) and v.shape == (T, HV, D)
    assert g.shape == (T, HV) and beta.shape == (T, HV)
    for x in (q, k, v):
        assert x.stride(1) == D and x.stride(2) == 1 and x.dtype == torch.bfloat16
    assert g.stride(1) == 1 and beta.stride(1) == 1
    assert state.shape == (HV, D, D) and state.dtype == torch.float32
    _, s_a, s_b = state.stride()
    s_k, s_v = (s_a, s_b) if state_kv else (s_b, s_a)
    gated = z is not None
    if out is None:
        out = torch.empty((T, HV * D) if gated else (T, HV, D), device=q.device, dtype=torch.bfloat16)
    assert out.stride(-1) == 1 and (not gated or z.stride(-1) == 1)

    NC = triton.cdiv(T, C)
    NC += NC % 2      # the scan takes chunks in pairs; a zero chunk leaves the state unchanged
    Tp = NC * C
    dev = state.device
    bf = torch.bfloat16
    w = torch.empty(Tp, HV, D, device=dev, dtype=bf)
    uT = torch.empty(NC, HV, D, C, device=dev, dtype=bf)
    kdT = torch.empty(NC, HV, D, C, device=dev, dtype=bf)
    G = torch.empty(HV, Tp, device=dev, dtype=torch.float32)
    qn = torch.empty(Tp, HK, D, device=dev, dtype=bf)
    M = torch.empty(NC, HV, C, C, device=dev, dtype=bf)
    wy(q, k, v, g, beta, w, uT, kdT, G, qn, M, T, Tp, NC, hip=WY_HIP)
    o = torch.empty(T, HV * D, device=dev, dtype=bf) if gated else out
    ck_every, ck = (0, None) if ckpt is None else ckpt
    if ck is not None:
        assert ck_every % C == 0 and ck.dtype == torch.float32 and ck.shape[1:] == state.shape
        assert ck.stride()[1:] == state.stride()
    rc = _lib.gdn_scan(SCAN_NW, w.data_ptr(), uT.data_ptr(), kdT.data_ptr(), qn.data_ptr(), M.data_ptr(),
                       G.data_ptr(), state.data_ptr(), state.stride(0), s_k, s_v, o.data_ptr(), o.stride(0), T, NC, Tp,
                       ck_every // C, 0 if ck is None else ck.shape[0], None if ck is None else ck.data_ptr(),
                       0 if ck is None else ck.stride(0), torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"gdn_scan: hip error {rc}")
    if gated:
        _gated_norm_kernel[(triton.cdiv(T, 32), HV)](o, z, gw, out, T, z.stride(0), out.stride(0),
                                                     BT=32, D=D, HV=HV, num_warps=4)
    return out
