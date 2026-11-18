"""Prefill attention experiments (not a test). Run: python3 -m eng.kernels.tune_prefill"""
import sys

import torch
import triton
import triton.language as tl

from . import attention as A
from .test_attention import make_cache, make_q, bench, prefill_flops, ref_attn, err, dev


@triton.jit
def _pf2(Q, K, V, O, G, p0, T, NBLK,
         sq_t, sq_h, sk_n, sk_h, so_t, so_h, sg_t, sg_h,
         qk_scale,
         GROUP: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
         HAS_GATE: tl.constexpr, REMAP: tl.constexpr, FMA: tl.constexpr, KTRANS: tl.constexpr,
         LAZY: tl.constexpr):
    if REMAP:
        # the 6 heads of a KV group and one row block on the same XCD (workgroups go round-robin over 8 XCDs)
        pid = tl.program_id(0)
        x = pid % 8
        slot = (pid // 8) % GROUP
        pair = pid // (8 * GROUP)
        h = (x % 4) * GROUP + slot
        r = pair * 2 + x // 4
        if r >= NBLK:
            return
        mb = NBLK - 1 - r
    else:
        h = tl.program_id(0)
        mb = NBLK - 1 - tl.program_id(1)
    g = h // GROUP
    start = mb * BLOCK_M
    offs_m = start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    n = p0 + T

    q = tl.load(Q + offs_m[:, None] * sq_t + h * sq_h + offs_d[None, :], mask=offs_m[:, None] < T, other=0.0)
    kT_ptr = K + g * sk_h + offs_d[:, None] + offs_n[None, :] * sk_n          # (D, BLOCK_N)
    k_ptr = K + g * sk_h + offs_n[:, None] * sk_n + offs_d[None, :]           # (BLOCK_N, D)
    v_ptr = V + g * sk_h + offs_n[:, None] * sk_n + offs_d[None, :]           # (BLOCK_N, D)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    full_end = ((p0 + start + 1) // BLOCK_N) * BLOCK_N
    for j in range(0, full_end, BLOCK_N):
        if KTRANS:
            kT = tl.trans(tl.load(k_ptr + j * sk_n))
        else:
            kT = tl.load(kT_ptr + j * sk_n)
        s = tl.dot(q, kT)
        if LAZY:
            # rescale only when a row max grows by more than 8 (log2): p <= 2^8 otherwise
            m_cand = tl.maximum(m_i, tl.max(s, 1) * qk_scale)
            upd = m_cand > m_i + 8.0
            m_new = tl.where(upd, m_cand, m_i)
            p = tl.math.exp2(s * qk_scale - m_new[:, None])
            v = tl.load(v_ptr + j * sk_n)
            if tl.max(upd.to(tl.int32), 0) > 0:
                alpha = tl.math.exp2(m_i - m_new)
                acc = acc * alpha[:, None]
                l_i = l_i * alpha
            l_i = l_i + tl.sum(p, 1)
            acc = acc + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        else:
            if FMA:
                m_new = tl.maximum(m_i, tl.max(s, 1) * qk_scale)
                p = tl.math.exp2(s * qk_scale - m_new[:, None])
            else:
                s = s * qk_scale
                m_new = tl.maximum(m_i, tl.max(s, 1))
                p = tl.math.exp2(s - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(v_ptr + j * sk_n)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
    hi = tl.minimum(p0 + start + BLOCK_M, n)
    pos = p0 + offs_m
    for j in range(full_end, hi, BLOCK_N):
        kn = j + offs_n
        if KTRANS:
            kT = tl.trans(tl.load(k_ptr + j * sk_n, mask=kn[:, None] < n, other=0.0))
        else:
            kT = tl.load(kT_ptr + j * sk_n, mask=kn[None, :] < n, other=0.0)
        s = tl.dot(q, kT) * qk_scale
        s = tl.where(kn[None, :] <= pos[:, None], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + j * sk_n, mask=kn[:, None] < n, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    if HAS_GATE:
        gt = tl.load(G + offs_m[:, None] * sg_t + h * sg_h + offs_d[None, :], mask=offs_m[:, None] < T, other=0.0)
        acc = acc * tl.sigmoid(gt.to(tl.float32))
    tl.store(O + offs_m[:, None] * so_t + h * so_h + offs_d[None, :], acc.to(O.dtype.element_ty),
             mask=offs_m[:, None] < T)


def pf2(q, k, v, p0, gate, cfg):
    T = q.shape[0]
    c = dict(cfg)
    BM, BN = c.pop("BLOCK_M"), c.pop("BLOCK_N")
    remap, fma, ktr, lazy = c.pop("REMAP", 0), c.pop("FMA", 0), c.pop("KTRANS", 0), c.pop("LAZY", 0)
    o = torch.empty(T, 24, 256, dtype=q.dtype, device=q.device)
    nblk = triton.cdiv(T, BM)
    grid = (triton.cdiv(nblk, 2) * 48,) if remap else (24, nblk)
    _pf2[grid](q, k, v, o, gate, p0, T, nblk, q.stride(0), q.stride(1), k.stride(0), k.stride(1), o.stride(0),
               o.stride(1), gate.stride(0), gate.stride(1), A.SCALE * A.LOG2E, GROUP=6, D=256, BLOCK_M=BM,
               BLOCK_N=BN, HAS_GATE=True, REMAP=remap, FMA=fma, KTRANS=ktr, LAZY=lazy, **c)
    return o


def main():
    base = dict(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=2, matrix_instr_nonkdim=16)
    variants = [dict(), dict(REMAP=1, FMA=1), dict(REMAP=1, LAZY=1), dict(LAZY=1),
                dict(REMAP=1, LAZY=1, waves_per_eu=2), dict(REMAP=1, LAZY=1, BLOCK_M=64),
                dict(REMAP=1, FMA=1, BLOCK_M=64), dict(REMAP=1, LAZY=1, BLOCK_M=64, num_warps=2),
                dict(REMAP=1, LAZY=1, matrix_instr_nonkdim=32), dict(REMAP=1, LAZY=1, BLOCK_N=32, num_stages=3)]
    for T, p0 in [(4096, 6400), (1500, 6400), (5600, 6400), (2048, 0)]:
        k, v = make_cache(p0 + T)
        q, gate = make_q(T)
        tf = prefill_flops(T, p0)
        ref = None
        if T in (1500, 2048):
            ref = ref_attn(q, k, v, p0 + T) * torch.sigmoid(gate.float())
        us = bench(lambda: A.attn_prefill(q, k, v, p0, gate=gate), iters=10, warm=2)
        print(f"T={T} p0={p0} current attn_prefill: {us:8.1f} us {tf / us / 1e6:6.1f} TFLOP/s", flush=True)
        for var in variants:
            c = dict(base, **var)
            try:
                us = bench(lambda: pf2(q, k, v, p0, gate, c), iters=10, warm=2)
                e = f"  err {err(pf2(q, k, v, p0, gate, c), ref)[0]:.1e}" if ref is not None else ""
                print(f"T={T} {var}: {us:8.1f} us {tf / us / 1e6:6.1f} TFLOP/s{e}", flush=True)
            except Exception as ex:
                print(f"T={T} {var}: failed {type(ex).__name__} {str(ex)[:150]}", flush=True)


if __name__ == "__main__":
    main()
