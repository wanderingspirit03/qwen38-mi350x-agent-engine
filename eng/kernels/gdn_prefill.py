"""Chunked gated delta rule forward for prefill (one sequence), in Triton.

gdn_prefill(q, k, v, g, beta, state, normalize_qk=True) -> o

  q, k:  (T, 16, 128) bf16. With normalize_qk=True they are the raw conv outputs: l2norm (eps 1e-6, fp32)
         is applied to both and q is multiplied by 128**-0.5 inside. With normalize_qk=False they must
         already be normalized (and q scaled).
  v:     (T, 48, 128) bf16.   g: (T, 48) fp32 log-decay (<= 0).   beta: (T, 48) fp32.
         Value head h uses key head h // 3. q, k, v, g, beta may be views with any token (row) stride, as
         long as the last dimension is contiguous and heads are 128 apart (q, k, v).
  state: (48, 128, 128) fp32, any strides; by default the transposed layout state[h, v, k] = S[h, k, v]
         (state_kv=False). Pass state_kv=True for state[h, k, v]. Updated in place to the state after T-1.
  Returns o: (T, 48, 128) bf16.

Semantics are those of eng.ref_ops.gdn_recurrent, computed by chunks of 64 positions:
  1. _wy_kernel, grid (chunks, 48): G = cumsum(g) inside the chunk; A = strictly lower part of
     beta_i k_i.k_j exp(G_i - G_j); X = (I + A)^-1 from 16x16 blocks (diagonal blocks by doubling, the rest
     by block substitution, fp32); w = X (beta exp(G) k), u = X (beta v), stored bf16.
  2. _scan_kernel, grid (48, 128 / BV), sequential over chunks, state block (128 x BV) fp32 in registers:
     h_c = S (stored bf16), vn = u - w S (stored bf16), S = exp(G_last) S + (exp(G_last - G) k)^T vn.
  3. _out_kernel, grid (chunks, 48): o = (exp(G) q) h_c + (q k^T * exp(G_i - G_j), j <= i) vn.
"""
import torch
import triton
import triton.language as tl

C = 64
D = 128
HK = 16
HV = 48


@triton.jit
def _sdot(a, b, SPLIT: tl.constexpr):
    """a @ b for fp32 a, b with bf16 MFMA. SPLIT=1: both rounded to bf16. SPLIT=2: b as bf16 hi + lo.
    SPLIT=3: both split, three products (close to fp32)."""
    ah = a.to(tl.bfloat16)
    bh = b.to(tl.bfloat16)
    acc = tl.dot(ah, bh)
    if SPLIT >= 2:
        acc += tl.dot(ah, (b - bh.to(tl.float32)).to(tl.bfloat16))
    if SPLIT >= 3:
        acc += tl.dot((a - ah.to(tl.float32)).to(tl.bfloat16), bh)
    return acc


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
def _load_k_block(q, k, qn_out, kn_out, r, m, hk, sq, sk, scale, STORE: tl.constexpr, NORM: tl.constexpr,
                  D: tl.constexpr, HK: tl.constexpr):
    d = tl.arange(0, D)
    kf = tl.load(k + r[:, None] * sk + hk * D + d[None, :], mask=m[:, None], other=0.).to(tl.float32)
    if NORM:
        kf = kf * tl.rsqrt(tl.sum(kf * kf, 1) + 1e-6)[:, None]
        if STORE:
            tl.store(kn_out + r[:, None] * (HK * D) + hk * D + d[None, :], kf.to(tl.bfloat16))
            qf = tl.load(q + r[:, None] * sq + hk * D + d[None, :], mask=m[:, None], other=0.).to(tl.float32)
            qf = qf * (tl.rsqrt(tl.sum(qf * qf, 1) + 1e-6) * scale)[:, None]
            tl.store(qn_out + r[:, None] * (HK * D) + hk * D + d[None, :], qf.to(tl.bfloat16))
    return kf.to(tl.bfloat16)


@triton.jit
def _block_A(ka, kb, ba, Ga, Gb, DIAG: tl.constexpr):
    A = tl.dot(ka, tl.trans(kb)) * ba[:, None]
    li = tl.arange(0, 16)
    if DIAG:
        A = A * tl.exp(tl.where(li[:, None] > li[None, :], Ga[:, None] - Gb[None, :], float("-inf")))
    else:
        A = A * tl.exp(Ga[:, None] - Gb[None, :])
    return A


@triton.jit(do_not_specialize=["T", "Tp"])
def _wy_kernel(q, k, v, g, beta, w_out, u_out, G_out, qn_out, kn_out,
               T, Tp, sq, sk, sv, sg, sb, scale,
               NORM: tl.constexpr, WU_SPLIT: tl.constexpr,
               D: tl.constexpr, HK: tl.constexpr, HV: tl.constexpr):
    c = tl.program_id(0)
    h = tl.program_id(1)
    hk = h // 3
    STORE_QK = NORM and (h % 3 == 0)
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

    k0 = _load_k_block(q, k, qn_out, kn_out, r0, m0, hk, sq, sk, scale, STORE_QK, NORM, D, HK)
    k1 = _load_k_block(q, k, qn_out, kn_out, r1, m1, hk, sq, sk, scale, STORE_QK, NORM, D, HK)
    k2 = _load_k_block(q, k, qn_out, kn_out, r2, m2, hk, sq, sk, scale, STORE_QK, NORM, D, HK)
    k3 = _load_k_block(q, k, qn_out, kn_out, r3, m3, hk, sq, sk, scale, STORE_QK, NORM, D, HK)

    g0 = tl.load(g + r0 * sg + h, mask=m0, other=0.)
    g1 = tl.load(g + r1 * sg + h, mask=m1, other=0.)
    g2 = tl.load(g + r2 * sg + h, mask=m2, other=0.)
    g3 = tl.load(g + r3 * sg + h, mask=m3, other=0.)
    G0 = tl.cumsum(g0, 0)
    G1 = tl.cumsum(g1, 0) + tl.sum(g0, 0)
    G2 = tl.cumsum(g2, 0) + (tl.sum(g0, 0) + tl.sum(g1, 0))
    G3 = tl.cumsum(g3, 0) + (tl.sum(g0, 0) + tl.sum(g1, 0) + tl.sum(g2, 0))
    tl.store(G_out + h * Tp + r0, G0)
    tl.store(G_out + h * Tp + r1, G1)
    tl.store(G_out + h * Tp + r2, G2)
    tl.store(G_out + h * Tp + r3, G3)
    b0 = tl.load(beta + r0 * sb + h, mask=m0, other=0.)
    b1 = tl.load(beta + r1 * sb + h, mask=m1, other=0.)
    b2 = tl.load(beta + r2 * sb + h, mask=m2, other=0.)
    b3 = tl.load(beta + r3 * sb + h, mask=m3, other=0.)

    A10 = _block_A(k1, k0, b1, G1, G0, False)
    A20 = _block_A(k2, k0, b2, G2, G0, False)
    A21 = _block_A(k2, k1, b2, G2, G1, False)
    A30 = _block_A(k3, k0, b3, G3, G0, False)
    A31 = _block_A(k3, k1, b3, G3, G1, False)
    A32 = _block_A(k3, k2, b3, G3, G2, False)
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

    # w = X (beta exp(G) k)
    x0 = k0.to(tl.float32) * (b0 * tl.exp(G0))[:, None]
    x1 = k1.to(tl.float32) * (b1 * tl.exp(G1))[:, None]
    x2 = k2.to(tl.float32) * (b2 * tl.exp(G2))[:, None]
    x3 = k3.to(tl.float32) * (b3 * tl.exp(G3))[:, None]
    wp = w_out + h * D + d[None, :]
    tl.store(wp + r0[:, None] * (HV * D), _sdot(T00, x0, WU_SPLIT).to(tl.bfloat16))
    tl.store(wp + r1[:, None] * (HV * D), (_sdot(T10, x0, WU_SPLIT) + _sdot(T11, x1, WU_SPLIT)).to(tl.bfloat16))
    tl.store(wp + r2[:, None] * (HV * D), (_sdot(T20, x0, WU_SPLIT) + _sdot(T21, x1, WU_SPLIT)
                                           + _sdot(T22, x2, WU_SPLIT)).to(tl.bfloat16))
    tl.store(wp + r3[:, None] * (HV * D), (_sdot(T30, x0, WU_SPLIT) + _sdot(T31, x1, WU_SPLIT)
                                           + _sdot(T32, x2, WU_SPLIT) + _sdot(T33, x3, WU_SPLIT)).to(tl.bfloat16))
    # u = X (beta v)
    vp = v + h * D + d[None, :]
    x0 = tl.load(vp + r0[:, None] * sv, mask=m0[:, None], other=0.).to(tl.float32) * b0[:, None]
    x1 = tl.load(vp + r1[:, None] * sv, mask=m1[:, None], other=0.).to(tl.float32) * b1[:, None]
    x2 = tl.load(vp + r2[:, None] * sv, mask=m2[:, None], other=0.).to(tl.float32) * b2[:, None]
    x3 = tl.load(vp + r3[:, None] * sv, mask=m3[:, None], other=0.).to(tl.float32) * b3[:, None]
    up = u_out + h * D + d[None, :]
    tl.store(up + r0[:, None] * (HV * D), _sdot(T00, x0, WU_SPLIT).to(tl.bfloat16))
    tl.store(up + r1[:, None] * (HV * D), (_sdot(T10, x0, WU_SPLIT) + _sdot(T11, x1, WU_SPLIT)).to(tl.bfloat16))
    tl.store(up + r2[:, None] * (HV * D), (_sdot(T20, x0, WU_SPLIT) + _sdot(T21, x1, WU_SPLIT)
                                           + _sdot(T22, x2, WU_SPLIT)).to(tl.bfloat16))
    tl.store(up + r3[:, None] * (HV * D), (_sdot(T30, x0, WU_SPLIT) + _sdot(T31, x1, WU_SPLIT)
                                           + _sdot(T32, x2, WU_SPLIT) + _sdot(T33, x3, WU_SPLIT)).to(tl.bfloat16))


@triton.jit
def _scan_load(wp, up, kp, Gp, c, T, sk, HV: tl.constexpr, D: tl.constexpr):
    r = c * 64 + tl.arange(0, 64)
    wt = tl.load(wp + r[:, None] * (HV * D))
    ut = tl.load(up + r[:, None] * (HV * D))
    kt = tl.load(kp + r[:, None] * sk, mask=(r < T)[:, None], other=0.)
    Gt = tl.load(Gp + r)
    Gl = tl.load(Gp + c * 64 + 63)
    return wt, ut, kt, Gt, Gl


@triton.jit
def _scan_step(S, c, wt, ut, kt, Gt, Gl, hp, vnp, S_SPLIT: tl.constexpr, HV: tl.constexpr, D: tl.constexpr):
    li = tl.arange(0, 64)
    tl.store(hp + c * (HV * D * D), S.to(tl.bfloat16))
    vn = ut.to(tl.float32) - _sdot(wt.to(tl.float32), S, S_SPLIT)
    tl.store(vnp + (c * 64 + li)[:, None] * (HV * D), vn.to(tl.bfloat16))
    kd = kt.to(tl.float32) * tl.exp(Gl - Gt)[:, None]
    return S * tl.exp(Gl) + _sdot(tl.trans(kd), vn, S_SPLIT)


@triton.jit(do_not_specialize=["T", "Tp", "NC"])
def _scan_kernel(k, w, u, G, state, hbuf, vn_out,
                 T, Tp, NC, sk, s_h, s_k, s_v,
                 S_SPLIT: tl.constexpr, BV: tl.constexpr,
                 D: tl.constexpr, HV: tl.constexpr):
    """NC is even (the buffers are padded to an even number of chunks). Two register sets: the loads of
    one chunk are in flight while the other chunk is processed."""
    h = tl.program_id(0)
    vb = tl.program_id(1)
    hk = h // 3
    d = tl.arange(0, D)
    dv = vb * BV + tl.arange(0, BV)
    sp = state + h * s_h + d[:, None] * s_k + dv[None, :] * s_v
    S = tl.load(sp)
    wp = w + h * D + d[None, :]
    up = u + h * D + dv[None, :]
    kp = k + hk * D + d[None, :]
    Gp = G + h * Tp
    hp = hbuf + h * (D * D) + d[:, None] * D + dv[None, :]
    vnp = vn_out + h * D + dv[None, :]

    wa, ua, ka, Ga, Gla = _scan_load(wp, up, kp, Gp, 0, T, sk, HV, D)
    for c in range(0, NC, 2):
        wb, ub, kb, Gb, Glb = _scan_load(wp, up, kp, Gp, c + 1, T, sk, HV, D)
        S = _scan_step(S, c, wa, ua, ka, Ga, Gla, hp, vnp, S_SPLIT, HV, D)
        wa, ua, ka, Ga, Gla = _scan_load(wp, up, kp, Gp, tl.minimum(c + 2, NC - 1), T, sk, HV, D)
        S = _scan_step(S, c + 1, wb, ub, kb, Gb, Glb, hp, vnp, S_SPLIT, HV, D)
    tl.store(sp, S)


@triton.jit(do_not_specialize=["T", "Tp"])
def _out_kernel(q, k, G, hbuf, vn, o,
                T, Tp, sq, sk,
                D: tl.constexpr, HV: tl.constexpr):
    c = tl.program_id(0)
    h = tl.program_id(1)
    hk = h // 3
    li = tl.arange(0, 64)
    d = tl.arange(0, D)
    r = c * 64 + li
    m = r < T
    qt = tl.load(q + r[:, None] * sq + hk * D + d[None, :], mask=m[:, None], other=0.)
    kt = tl.load(k + r[:, None] * sk + hk * D + d[None, :], mask=m[:, None], other=0.)
    Gt = tl.load(G + h * Tp + r)
    att = tl.dot(qt, tl.trans(kt))
    att = att * tl.exp(tl.where(li[:, None] >= li[None, :], Gt[:, None] - Gt[None, :], float("-inf")))
    ht = tl.load(hbuf + (c * HV + h) * (D * D) + d[:, None] * D + d[None, :])
    vt = tl.load(vn + r[:, None] * (HV * D) + h * D + d[None, :])
    qg = (qt.to(tl.float32) * tl.exp(Gt)[:, None]).to(tl.bfloat16)
    ot = tl.dot(qg, ht) + tl.dot(att.to(tl.bfloat16), vt)
    tl.store(o + r[:, None] * (HV * D) + h * D + d[None, :], ot.to(o.dtype.element_ty), mask=m[:, None])


def gdn_prefill(q, k, v, g, beta, state, normalize_qk=True, state_kv=False, out=None,
                BV=16, wu_split=1, s_split=1, wy_warps=1, scan_warps=2, out_warps=2):
    """See the module docstring. The keyword arguments after `out` are tuning knobs; the defaults are the
    fastest measured on MI350X."""
    T = q.shape[0]
    assert q.shape == (T, HK, D) and k.shape == (T, HK, D) and v.shape == (T, HV, D)
    assert g.shape == (T, HV) and beta.shape == (T, HV) and state.shape == (HV, D, D)
    for x in (q, k, v):
        assert x.stride(1) == D and x.stride(2) == 1 and x.dtype == torch.bfloat16
    assert g.stride(1) == 1 and beta.stride(1) == 1 and state.dtype == torch.float32
    NC = triton.cdiv(T, C)
    NCp = NC + NC % 2          # the scan handles chunks in pairs
    Tp = NCp * C
    dev = q.device
    w = torch.empty(Tp, HV, D, device=dev, dtype=torch.bfloat16)
    u = torch.empty(Tp, HV, D, device=dev, dtype=torch.bfloat16)
    G = torch.empty(HV, Tp, device=dev, dtype=torch.float32)
    if normalize_qk:
        qn = torch.empty(Tp, HK, D, device=dev, dtype=torch.bfloat16)
        kn = torch.empty(Tp, HK, D, device=dev, dtype=torch.bfloat16)
    else:
        qn, kn = q, k
    _wy_kernel[(NCp, HV)](q, k, v, g, beta, w, u, G, qn, kn,
                          T, Tp, q.stride(0), k.stride(0), v.stride(0), g.stride(0), beta.stride(0), D ** -0.5,
                          NORM=normalize_qk, WU_SPLIT=wu_split, D=D, HK=HK, HV=HV, num_warps=wy_warps)
    hbuf = torch.empty(NCp, HV, D, D, device=dev, dtype=torch.bfloat16)
    vn = torch.empty(Tp, HV, D, device=dev, dtype=torch.bfloat16)
    s_h, s_a, s_b = state.stride()
    s_k, s_v = (s_a, s_b) if state_kv else (s_b, s_a)
    _scan_kernel[(HV, D // BV)](kn, w, u, G, state, hbuf, vn, T, Tp, NCp, kn.stride(0), s_h, s_k, s_v,
                                S_SPLIT=s_split, BV=BV, D=D, HV=HV, num_warps=scan_warps)
    o = torch.empty(T, HV, D, device=dev, dtype=torch.bfloat16) if out is None else out
    _out_kernel[(NC, HV)](qn, kn, G, hbuf, vn, o, T, Tp, qn.stride(0), kn.stride(0),
                          D=D, HV=HV, num_warps=out_warps)
    return o
