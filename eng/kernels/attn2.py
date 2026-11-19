"""Prefill attention with 8-bit matmuls (attn2): causal flash attention for T new positions p0..p0+T-1.

attn2_prefill(q, k_cache, v_cache, p0, gate=None, out=None) -> o     same contract as attention.attn_prefill

Numerics (per query row, keys 0..p0+i):
- keys 0..63 (attention sinks) and the keys of the diagonal blocks plus RECENT * 64 keys before them (local attention,
  where most of the weight is) run in bf16 from the cache, like attention.attn_prefill;
- all other keys: Q.K in int8 (Q: one scale per 128-row tile, K: one scale per 64 keys and head) or fp8 e4m3 (QK8=False),
  P.V in fp8 e4m3 (P = 2^(s - m + 8) <= 256, V / V_SCALE); fp32 accumulation and softmax.
A quantization pass writes the 8-bit copies of keys/values 0..p0+T-1 first (~16 us at 12k positions).
"""
import torch
import triton
import triton.language as tl

HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
GROUP = HEADS // KV_HEADS
SCALE = HEAD_DIM ** -0.5
LOG2E = 1.4426950408889634
V_SCALE = 0.5           # fixed fp8 scale of V: |v| <= 224 kept, larger values clamped (seen: <= 117)
K_SCALE = 1.0           # fixed fp8 scale of K for QK8=False (keys are RMS-normed, |k| <~ 25 seen)
BLOCK_M, BLOCK_N = 128, 64


# ------------------------------------------------------------------------------------------- quantization pass
@triton.jit(do_not_specialize=["n"])
def _kv_quant_kernel(K, V, KQ, VQ, KS, n, sk_n, v_inv, k_inv, KVH: tl.constexpr, D: tl.constexpr,
                     BN: tl.constexpr, QK8: tl.constexpr):
    # one program per (BN-key block, head). K -> int8 with one scale per block (KS[b, g]) or fp8 (k * k_inv);
    # V -> fp8 e4m3 (v * v_inv, clamped). Rows >= n are written as zeros.
    b = tl.program_id(0)
    g = tl.program_id(1)
    r = b * BN + tl.arange(0, BN)
    c = tl.arange(0, D)
    m = r[:, None] < n
    k = tl.load(K + r[:, None] * sk_n + g * D + c[None, :], mask=m, other=0.0).to(tl.float32)
    v = tl.load(V + r[:, None] * sk_n + g * D + c[None, :], mask=m, other=0.0).to(tl.float32) * v_inv
    if QK8:
        amax = tl.maximum(tl.max(tl.max(tl.abs(k), 1), 0), 1e-20)
        kq = tl.floor(k * (127.0 / amax) + 0.5).to(tl.int8)
        tl.store(KS + b * KVH + g, amax / 127.0)
    else:
        kq = tl.minimum(tl.maximum(k * k_inv, -448.0), 448.0).to(KQ.dtype.element_ty)
    tl.store(KQ + r[:, None] * (KVH * D) + g * D + c[None, :], kq)
    v = tl.minimum(tl.maximum(v, -448.0), 448.0)
    tl.store(VQ + r[:, None] * (KVH * D) + g * D + c[None, :], v.to(VQ.dtype.element_ty))


_qbuf = {}


def quant_kv(k_cache, v_cache, n, QK8=True):
    """8-bit copies of keys/values 0..n-1, zero up to a multiple of 128 rows. Returns kq, vq, ks (K block scales)."""
    dev = k_cache.device
    lp = triton.cdiv(n, 128) * 128
    key = (dev, QK8)
    if key not in _qbuf or _qbuf[key][0].shape[0] < lp:
        lp = max(lp, triton.cdiv(k_cache.shape[0], 128) * 128)
        _qbuf[key] = (torch.empty(lp, KV_HEADS, HEAD_DIM, dtype=torch.int8 if QK8 else torch.float8_e4m3fn, device=dev),
                      torch.empty(lp, KV_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev),
                      torch.empty(lp // BLOCK_N, KV_HEADS, dtype=torch.float32, device=dev))
    kq, vq, ks = _qbuf[key]
    assert k_cache.stride(1) == HEAD_DIM and k_cache.stride(2) == 1 and k_cache.stride() == v_cache.stride()
    nb = triton.cdiv(n, 128) * (128 // BLOCK_N)
    _kv_quant_kernel[(nb, KV_HEADS)](k_cache, v_cache, kq, vq, ks, n, k_cache.stride(0), 1.0 / V_SCALE,
                                     1.0 / K_SCALE, KVH=KV_HEADS, D=HEAD_DIM, BN=BLOCK_N, QK8=QK8, num_warps=4)
    return kq, vq, ks


# ------------------------------------------------------------------------------------------- attention
@triton.jit
def _blk16(q16, k16_ptr, v16_ptr, j, n, pos, acc, m_i, l_i, sc_n, qk_scale16, inv_vs, BLOCK_N: tl.constexpr):
    # key block j in bf16 from the cache, causal mask; rows that have seen no key keep m = -inf, l = 0
    kn = j * BLOCK_N + tl.arange(0, BLOCK_N)
    kT = tl.load(k16_ptr + j * BLOCK_N * sc_n, mask=kn[None, :] < n, other=0.0)
    s = tl.dot(q16, kT) * qk_scale16
    s = tl.where(kn[None, :] <= pos[:, None], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = tl.math.exp2(s - (m_use - 8.0)[:, None])
    alpha = tl.math.exp2(m_i - m_use)
    l_i = l_i * alpha + tl.sum(p, 1)
    v = tl.load(v16_ptr + j * BLOCK_N * sc_n, mask=kn[:, None] < n, other=0.0)
    acc = acc * alpha[:, None] + tl.dot((p * inv_vs).to(tl.bfloat16), v)
    return acc, m_new, l_i


@triton.jit
def _item(Q, K, V, KS, K16, V16, h, start, kb0, kb1, p0, T,
          sq_t, sq_h, sk_n, sk_h, sc_n, sc_h, qk_scale, v_scale, qk_scale16,
          GROUP: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
          QK8: tl.constexpr, FIRST16: tl.constexpr, RECENT: tl.constexpr, STAGES: tl.constexpr):
    """Key blocks kb0..kb1-1 for rows start..start+BLOCK_M-1 of query head h. Returns acc (in units of V / v_scale,
    not normalized), m (log2 units), l (both with the 2^8 factor of P)."""
    g = h // GROUP
    n = p0 + T
    offs_m = start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    pos = p0 + offs_m
    inv_vs = 1.0 / v_scale
    q_ptr = Q + offs_m[:, None] * sq_t + h * sq_h + offs_d[None, :]
    k16_ptr = K16 + g * sc_h + offs_d[:, None] + offs_n[None, :] * sc_n
    v16_ptr = V16 + g * sc_h + offs_n[:, None] * sc_n + offs_d[None, :]
    # block ranges: [0, j0b) bf16 (sinks), [j0b, qe_b) 8-bit (all keys visible), [qe_b, ...) bf16 with causal mask
    j0b = 1 if FIRST16 else 0
    qe_b = tl.maximum((p0 + start + 1) // BLOCK_N - RECENT, j0b)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    for jb in range(kb0, tl.minimum(kb1, j0b)):          # at most one block: the sinks
        q16 = tl.load(q_ptr, mask=offs_m[:, None] < T, other=0.0)
        acc, m_i, l_i = _blk16(q16, k16_ptr, v16_ptr, jb, n, pos, acc, m_i, l_i, sc_n, qk_scale16, inv_vs, BLOCK_N)

    qf = tl.load(q_ptr, mask=offs_m[:, None] < T, other=0.0).to(tl.float32)
    qmax = tl.maximum(tl.max(tl.max(tl.abs(qf), 1), 0), 1e-20)
    if QK8:
        q = tl.floor(qf * (127.0 / qmax) + 0.5).to(tl.int8)
        fq = qmax / 127.0 * qk_scale
    else:
        q = (qf * (448.0 / qmax)).to(tl.float8e4nv)
        fq = qmax / 448.0 * qk_scale
    kT_ptr = K + g * sk_h + offs_d[:, None] + offs_n[None, :] * sk_n
    v_ptr = V + g * sk_h + offs_n[:, None] * sk_n + offs_d[None, :]
    for jb in tl.range(tl.maximum(kb0, j0b), tl.minimum(kb1, qe_b), num_stages=STAGES):
        j = jb * BLOCK_N
        kT = tl.load(kT_ptr + j * sk_n)
        if QK8:
            s = tl.dot(q, kT, out_dtype=tl.int32).to(tl.float32)
            f = fq * tl.load(KS + jb * KVH + g)
        else:
            s = tl.dot(q, kT)
            f = fq
        m_new = tl.maximum(m_i, tl.max(s, 1) * f)
        p = tl.math.exp2(s * f - (m_new - 8.0)[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + j * sk_n)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    q16 = tl.load(q_ptr, mask=offs_m[:, None] < T, other=0.0)
    for jb in tl.range(tl.maximum(kb0, qe_b), kb1, num_stages=2):
        acc, m_i, l_i = _blk16(q16, k16_ptr, v16_ptr, jb, n, pos, acc, m_i, l_i, sc_n, qk_scale16, inv_vs, BLOCK_N)
    return acc, m_i, l_i


@triton.jit
def _store(O, G, acc, start, h, T, so_t, so_h, sg_t, sg_h, BLOCK_M: tl.constexpr, D: tl.constexpr,
           HAS_GATE: tl.constexpr):
    offs_m = start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    if HAS_GATE:
        gt = tl.load(G + offs_m[:, None] * sg_t + h * sg_h + offs_d[None, :], mask=offs_m[:, None] < T, other=0.0)
        acc = acc * tl.sigmoid(gt.to(tl.float32))
    tl.store(O + offs_m[:, None] * so_t + h * so_h + offs_d[None, :], acc.to(O.dtype.element_ty),
             mask=offs_m[:, None] < T)


@triton.jit
def _nblocks(mb, p0, n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    return (tl.minimum(p0 + (mb + 1) * BLOCK_M, n) + BLOCK_N - 1) // BLOCK_N


@triton.jit(do_not_specialize=["p0", "T", "NBLK"])
def _attn2_kernel(Q, K, V, KS, K16, V16, O, G, p0, T, NBLK,
                  sq_t, sq_h, sk_n, sk_h, sc_n, sc_h, so_t, so_h, sg_t, sg_h,
                  qk_scale, v_scale, qk_scale16,
                  GROUP: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr, HAS_GATE: tl.constexpr, QK8: tl.constexpr, FIRST16: tl.constexpr,
                  RECENT: tl.constexpr, STAGES: tl.constexpr):
    # 1-D grid of 48 programs per pair of row blocks, longest rows first; XCD x gets the 6 query heads of KV head
    # x % 4 for one row block, so they share K/V in that XCD's L2
    pid = tl.program_id(0)
    x = pid % 8
    h = (x % 4) * GROUP + (pid // 8) % GROUP
    r = (pid // (8 * GROUP)) * 2 + x // 4
    if r >= NBLK:
        return
    start = (NBLK - 1 - r) * BLOCK_M
    nb = _nblocks(NBLK - 1 - r, p0, p0 + T, BLOCK_M, BLOCK_N)
    acc, m_i, l_i = _item(Q, K, V, KS, K16, V16, h, start, 0, nb, p0, T,
                          sq_t, sq_h, sk_n, sk_h, sc_n, sc_h, qk_scale, v_scale, qk_scale16,
                          GROUP, KVH, D, BLOCK_M, BLOCK_N, QK8, FIRST16, RECENT, STAGES)
    acc = acc * (v_scale / l_i[:, None])
    _store(O, G, acc, start, h, T, so_t, so_h, sg_t, sg_h, BLOCK_M, D, HAS_GATE)


def use_attn2(T, p0):
    """Where attn2 is faster than attention.attn_prefill (measured, MI350X): enough rows and enough keys outside the
    bf16 window. Small chunks keep the bf16 kernel (its stream-K schedule fills the GPU better)."""
    return T >= 768 and p0 + T >= 4096


def attn2_prefill(q, k_cache, v_cache, p0, gate=None, out=None, QK8=True, FIRST16=True, RECENT=8, STAGES=2,
                  num_warps=4, BLOCK_M=BLOCK_M, **extra):
    """q (T, 24, 256) bf16 at positions p0..p0+T-1 (may be strided, last dim contiguous); k_cache/v_cache
    (L, 4, 256) bf16 contiguous with positions < p0 + T written. Returns o (T, 24, 256) bf16 (* sigmoid(gate))."""
    T = q.shape[0]
    assert q.shape[1:] == (HEADS, HEAD_DIM) and q.stride(2) == 1
    o = torch.empty(T, HEADS, HEAD_DIM, dtype=q.dtype, device=q.device) if out is None else out
    gt = gate if gate is not None else o
    p0 = int(p0)
    n = p0 + T
    kq, vq, ks = quant_kv(k_cache, v_cache, n, QK8)
    nblk = triton.cdiv(T, BLOCK_M)
    _attn2_kernel[(triton.cdiv(nblk, 2) * 2 * HEADS,)](
        q, kq, vq, ks, k_cache, v_cache, o, gt, p0, T, nblk,
        q.stride(0), q.stride(1), kq.stride(0), kq.stride(1), k_cache.stride(0), k_cache.stride(1),
        o.stride(0), o.stride(1), gt.stride(0), gt.stride(1),
        SCALE * LOG2E * (1.0 if QK8 else K_SCALE), V_SCALE, SCALE * LOG2E,
        GROUP=GROUP, KVH=KV_HEADS, D=HEAD_DIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HAS_GATE=gate is not None,
        QK8=QK8, FIRST16=FIRST16, RECENT=RECENT, STAGES=STAGES, num_warps=num_warps, **extra)
    return o
