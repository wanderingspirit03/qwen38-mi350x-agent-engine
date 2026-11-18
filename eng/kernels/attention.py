"""Full-attention kernels for Qwen3.8-27B (24 query heads, 4 KV heads, head dim 256), batch 1.

attn_prefill(q, k_cache, v_cache, p0)  -> o   causal flash attention for T new positions p0..p0+T-1
                                              (stream-K schedule when few row blocks see many keys)
attn_decode(q, k_cache, v_cache, n)    -> o   split-K attention for T <= 16 new positions n-T..n-1;
                                              n is a device int32 tensor (CUDA/HIP-graph safe)

q, o: (T, 24, 256) bf16 (q may be strided; last dim contiguous). k_cache, v_cache: (L_max, 4, 256) bf16
with the keys/values of all positions < p0 + T (prefill) or < n (decode) already written.
Query head h uses KV head h // 6; scale 1/16. Optional `gate` (T, 24, 256): o *= sigmoid(gate) (fused).
"""
import functools

import torch
import triton
import triton.language as tl

HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
GROUP = HEADS // KV_HEADS
SCALE = HEAD_DIM ** -0.5
LOG2E = 1.4426950408889634


# ----------------------------------------------------------------------------------------------- prefill
@triton.jit
def _prefill_kernel(Q, K, V, O, G, p0, T, NBLK,
                    sq_t, sq_h, sk_n, sk_h, so_t, so_h, sg_t, sg_h,
                    qk_scale,
                    GROUP: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    HAS_GATE: tl.constexpr):
    # 1-D grid of 48 programs per pair of row blocks, longest rows first. Workgroups go round-robin over the 8 XCDs:
    # XCD x gets the 6 query heads of KV head x % 4 for one row block, so they share K/V in that XCD's L2.
    pid = tl.program_id(0)
    x = pid % 8
    h = (x % 4) * GROUP + (pid // 8) % GROUP
    r = (pid // (8 * GROUP)) * 2 + x // 4
    if r >= NBLK:
        return
    mb = NBLK - 1 - r
    g = h // GROUP
    start = mb * BLOCK_M
    offs_m = start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    n = p0 + T

    q = tl.load(Q + offs_m[:, None] * sq_t + h * sq_h + offs_d[None, :], mask=offs_m[:, None] < T, other=0.0)
    kT_ptr = K + g * sk_h + offs_d[:, None] + offs_n[None, :] * sk_n          # (D, BLOCK_N)
    v_ptr = V + g * sk_h + offs_n[:, None] * sk_n + offs_d[None, :]           # (BLOCK_N, D)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    # keys every row of the block sees: j <= p0 + start
    full_end = ((p0 + start + 1) // BLOCK_N) * BLOCK_N
    for j in range(0, full_end, BLOCK_N):
        kT = tl.load(kT_ptr + j * sk_n)
        s = tl.dot(q, kT)                       # unscaled: the scale goes into one FMA with the max
        m_new = tl.maximum(m_i, tl.max(s, 1) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + j * sk_n)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    # diagonal part: causal mask
    hi = tl.minimum(p0 + start + BLOCK_M, n)
    pos = p0 + offs_m
    for j in range(full_end, hi, BLOCK_N):
        kn = j + offs_n
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


@triton.jit
def _nblocks(mb, p0, n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    return (tl.minimum(p0 + (mb + 1) * BLOCK_M, n) + BLOCK_N - 1) // BLOCK_N


@triton.jit
def _prefill_sk_kernel(Q, K, V, O, G, WO, WL, p0, T, NBLK, KTOT, NPROG,
           sq_t, sq_h, sk_n, sk_h, so_t, so_h, sg_t, sg_h,
           qk_scale,
           GROUP: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
           HAS_GATE: tl.constexpr):
    # Stream-K: the work is KTOT (item, key block) iterations, items = (row block, head), row blocks longest first,
    # heads in order. Program pid does iterations [lo, hi); an item it does only in part goes to the workspace
    # (slot 0: its first item, slot 1: its last), merged by _prefill_merge_kernel.
    pid = tl.program_id(0)
    lo = pid * KTOT // NPROG
    hi = (pid + 1) * KTOT // NPROG
    n = p0 + T
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    r = 0
    base = 0
    nb = _nblocks(NBLK - 1, p0, n, BLOCK_M, BLOCK_N)
    while base + 24 * nb <= lo:
        base += 24 * nb
        r += 1
        nb = _nblocks(NBLK - 1 - r, p0, n, BLOCK_M, BLOCK_N)
    it = lo
    while it < hi:
        h = (it - base) // nb
        ibase = base + h * nb
        kb0 = it - ibase
        kb1 = tl.minimum(hi - ibase, nb)
        mb = NBLK - 1 - r
        g = h // GROUP
        start = mb * BLOCK_M
        offs_m = start + tl.arange(0, BLOCK_M)
        q = tl.load(Q + offs_m[:, None] * sq_t + h * sq_h + offs_d[None, :], mask=offs_m[:, None] < T, other=0.0)
        kT_ptr = K + g * sk_h + offs_d[:, None] + offs_n[None, :] * sk_n
        v_ptr = V + g * sk_h + offs_n[:, None] * sk_n + offs_d[None, :]
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)
        nfull = (p0 + start + 1) // BLOCK_N
        for j in range(kb0 * BLOCK_N, tl.minimum(kb1, nfull) * BLOCK_N, BLOCK_N):
            kT = tl.load(kT_ptr + j * sk_n)
            s = tl.dot(q, kT)
            m_new = tl.maximum(m_i, tl.max(s, 1) * qk_scale)
            p = tl.math.exp2(s * qk_scale - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(v_ptr + j * sk_n)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        pos = p0 + offs_m
        for j in range(tl.maximum(kb0, nfull) * BLOCK_N, kb1 * BLOCK_N, BLOCK_N):
            kn = j + offs_n
            kT = tl.load(kT_ptr + j * sk_n, mask=kn[None, :] < n, other=0.0)
            s = tl.dot(q, kT) * qk_scale
            s = tl.where(kn[None, :] <= pos[:, None], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_use = tl.where(m_new == float("-inf"), 0.0, m_new)    # rows that have seen no key yet
            p = tl.math.exp2(s - m_use[:, None])
            alpha = tl.math.exp2(m_i - m_use)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(v_ptr + j * sk_n, mask=kn[:, None] < n, other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        acc = acc / tl.maximum(l_i, 1e-30)[:, None]    # a part may hold rows that see none of its keys
        if (kb0 == 0) & (kb1 == nb):
            if HAS_GATE:
                gt = tl.load(G + offs_m[:, None] * sg_t + h * sg_h + offs_d[None, :], mask=offs_m[:, None] < T,
                             other=0.0)
                acc = acc * tl.sigmoid(gt.to(tl.float32))
            tl.store(O + offs_m[:, None] * so_t + h * so_h + offs_d[None, :], acc.to(O.dtype.element_ty),
                     mask=offs_m[:, None] < T)
        else:
            slot = pid * 2 + (it != lo).to(tl.int32)
            rows = tl.arange(0, BLOCK_M)
            tl.store(WO + (slot * BLOCK_M + rows[:, None]) * D + offs_d[None, :], acc)
            tl.store(WL + slot * BLOCK_M + rows, m_i + tl.math.log2(l_i))
        it = ibase + kb1
        if it == base + 24 * nb:
            base += 24 * nb
            r += 1
            nb = _nblocks(NBLK - 1 - r, p0, n, BLOCK_M, BLOCK_N)


@triton.jit
def _prefill_merge_kernel(O, G, WO, WL, p0, T, NBLK, KTOT, NPROG, so_t, so_h, sg_t, sg_h,
              D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HAS_GATE: tl.constexpr):
    # one program per item; items done by one program only were written by _prefill_sk_kernel
    i = tl.program_id(0)
    r = i // 24
    h = i % 24
    n = p0 + T
    base = 0
    for rr in range(0, r):
        base += 24 * _nblocks(NBLK - 1 - rr, p0, n, BLOCK_M, BLOCK_N)
    nb = _nblocks(NBLK - 1 - r, p0, n, BLOCK_M, BLOCK_N)
    s0 = base + h * nb
    s1 = s0 + nb - 1
    # program containing iteration x: largest p with p * KTOT // NPROG <= x
    pf = (s0 * NPROG + NPROG - 1) // KTOT
    pf = tl.where(pf * KTOT // NPROG > s0, pf - 1, pf)
    pf = tl.where((pf + 1) * KTOT // NPROG <= s0, pf + 1, pf)
    pl = (s1 * NPROG + NPROG - 1) // KTOT
    pl = tl.where(pl * KTOT // NPROG > s1, pl - 1, pl)
    pl = tl.where((pl + 1) * KTOT // NPROG <= s1, pl + 1, pl)
    if pf == pl:
        return
    rows = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m = tl.full([BLOCK_M], float("-inf"), tl.float32)
    for p in range(pf, pl + 1):
        slot = p * 2 + (p * KTOT // NPROG < s0).to(tl.int32)
        m = tl.maximum(m, tl.load(WL + slot * BLOCK_M + rows))
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    den = tl.zeros([BLOCK_M], tl.float32)
    for p in range(pf, pl + 1):
        slot = p * 2 + (p * KTOT // NPROG < s0).to(tl.int32)
        w = tl.math.exp2(tl.load(WL + slot * BLOCK_M + rows) - m)
        den += w
        acc += w[:, None] * tl.load(WO + (slot * BLOCK_M + rows[:, None]) * D + offs_d[None, :])
    acc = acc / den[:, None]
    offs_m = (NBLK - 1 - r) * BLOCK_M + rows
    if HAS_GATE:
        gt = tl.load(G + offs_m[:, None] * sg_t + h * sg_h + offs_d[None, :], mask=offs_m[:, None] < T, other=0.0)
        acc = acc * tl.sigmoid(gt.to(tl.float32))
    tl.store(O + offs_m[:, None] * so_t + h * so_h + offs_d[None, :], acc.to(O.dtype.element_ty),
             mask=offs_m[:, None] < T)


def prefill_cfg(T):
    if T <= 1024:          # few row blocks: smaller blocks give more programs
        return dict(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2)
    return dict(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=2, matrix_instr_nonkdim=16)


@functools.cache
def _num_cus(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


_sk_ws = {}


def _use_stream_k(T, p0, ncu):
    # measured: stream-K wins (up to 2.5x) when ceil(T / 128) * 24 items do not fill the CUs much more than once and
    # the keys are long; it loses on short keys (its merge and workspace cost more than the imbalance)
    return p0 >= 2048 and triton.cdiv(T, 128) * HEADS <= 1.2 * ncu


def attn_prefill(q, k_cache, v_cache, p0, gate=None, out=None, cfg=None):
    """q (T, 24, 256) bf16 at positions p0..p0+T-1; returns o (T, 24, 256) bf16."""
    T = q.shape[0]
    assert q.shape[1:] == (HEADS, HEAD_DIM) and q.stride(2) == 1 and k_cache.stride(2) == 1
    assert k_cache.stride() == v_cache.stride()
    o = torch.empty(T, HEADS, HEAD_DIM, dtype=q.dtype, device=q.device) if out is None else out
    gt = gate if gate is not None else o
    p0 = int(p0)
    ncu = _num_cus(q.device)
    if cfg is None and _use_stream_k(T, p0, ncu):
        BM, BN = 128, 64
        nblk = triton.cdiv(T, BM)
        n = p0 + T
        ktot = HEADS * sum(triton.cdiv(min(p0 + (mb + 1) * BM, n), BN) for mb in range(nblk))
        if q.device not in _sk_ws:
            _sk_ws[q.device] = (torch.empty(ncu * 2, BM, HEAD_DIM, dtype=torch.float32, device=q.device),
                                torch.empty(ncu * 2, BM, dtype=torch.float32, device=q.device))
        wo, wl = _sk_ws[q.device]
        _prefill_sk_kernel[(ncu,)](
            q, k_cache, v_cache, o, gt, wo, wl, p0, T, nblk, ktot, ncu,
            q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), o.stride(0), o.stride(1),
            gt.stride(0), gt.stride(1), SCALE * LOG2E, GROUP=GROUP, D=HEAD_DIM, BLOCK_M=BM, BLOCK_N=BN,
            HAS_GATE=gate is not None, num_warps=4, num_stages=2, matrix_instr_nonkdim=16)
        _prefill_merge_kernel[(nblk * HEADS,)](
            o, gt, wo, wl, p0, T, nblk, ktot, ncu, o.stride(0), o.stride(1), gt.stride(0), gt.stride(1),
            D=HEAD_DIM, BLOCK_M=BM, BLOCK_N=BN, HAS_GATE=gate is not None, num_warps=8)
        return o
    cfg = dict(prefill_cfg(T), **(cfg or {}))
    nblk = triton.cdiv(T, cfg["BLOCK_M"])
    extra = {k: v for k, v in cfg.items() if k not in ("BLOCK_M", "BLOCK_N")}
    _prefill_kernel[(triton.cdiv(nblk, 2) * 2 * HEADS,)](
        q, k_cache, v_cache, o, gt, p0, T, nblk,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), o.stride(0), o.stride(1),
        gt.stride(0), gt.stride(1), SCALE * LOG2E,
        GROUP=GROUP, D=HEAD_DIM, BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"],
        HAS_GATE=gate is not None, **extra)
    return o


# ----------------------------------------------------------------------------------------------- decode
# Stage 1: grid (NSPLIT, 4 KV heads). The rows of one program are the GROUP*T (query head, token) pairs of one
# KV head (row r = t * GROUP + hh), so each K/V byte is read once. The key range [0, n) is cut into NSPLIT
# chunks (a multiple of BLOCK_N long, sized from n on the device); programs past the end exit.
# Stage 2: grid (GROUP*T, 4): merges the chunk partials with their log-sum-exp.

@triton.jit
def _decode_split_kernel(Q, K, V, NPTR, OP, LSE, T,
                         sq_t, sq_h, sk_n, sk_h,
                         qk_scale,
                         GROUP: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         NSPLIT: tl.constexpr, NRB: tl.constexpr, ROWS: tl.constexpr):
    # program -> (split, row block): the NRB row blocks of one split get program ids 8 apart, so they are
    # dispatched to the same XCD (workgroups go round-robin over the 8 XCDs) and share its L2.
    pid = tl.program_id(0)
    s_id = (pid // (8 * NRB)) * 8 + pid % 8
    rb = (pid // 8) % NRB
    g = tl.program_id(1)
    offs_r = rb * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = offs_r < T * GROUP
    base = (s_id * tl.num_programs(1) + g) * ROWS
    n = tl.load(NPTR)
    chunk = tl.cdiv(tl.cdiv(n, NSPLIT), BLOCK_N) * BLOCK_N
    lo = s_id * chunk
    if lo >= n:
        # unused split: weight 0 in the merge (its partial output stays finite: the workspace starts zeroed)
        tl.store(LSE + base + offs_r, tl.full([BLOCK_M], float("-inf"), tl.float32), mask=rmask)
        return
    hi = tl.minimum(lo + chunk, n)

    t = offs_r // GROUP
    hh = offs_r % GROUP
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + t[:, None] * sq_t + (g * GROUP + hh)[:, None] * sq_h + offs_d[None, :],
                mask=rmask[:, None], other=0.0)
    pos = n - T + t                                   # padded rows (t >= T) see everything; never stored
    kT_ptr = K + g * sk_h + offs_d[:, None] + offs_n[None, :] * sk_n
    v_ptr = V + g * sk_h + offs_n[:, None] * sk_n + offs_d[None, :]

    m_i = tl.full([BLOCK_M], -1e30, tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    for j in range(lo, hi, BLOCK_N):
        kn = j + offs_n
        kT = tl.load(kT_ptr + j * sk_n, mask=kn[None, :] < hi, other=0.0)
        v = tl.load(v_ptr + j * sk_n, mask=kn[:, None] < hi, other=0.0)
        s = tl.dot(q, kT) * qk_scale
        s = tl.where((kn[None, :] <= pos[:, None]) & (kn[None, :] < hi), s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    lse = tl.where(l_i > 0, m_i + tl.math.log2(l_safe), float("-inf"))
    tl.store(LSE + base + offs_r, lse, mask=rmask)
    tl.store(OP + (base + offs_r)[:, None] * D + offs_d[None, :], (acc / l_safe[:, None]).to(OP.dtype.element_ty),
             mask=rmask[:, None])


@triton.jit
def _decode_merge_kernel(OP, LSE, O, G,
                         so_t, so_h, sg_t, sg_h,
                         GROUP: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr,
                         NSPLIT: tl.constexpr, NSPLIT_P2: tl.constexpr, HAS_GATE: tl.constexpr):
    r = tl.program_id(0)
    g = tl.program_id(1)
    offs_s = tl.arange(0, NSPLIT_P2)
    offs_d = tl.arange(0, D)
    smask = offs_s < NSPLIT
    row = (offs_s * tl.num_programs(1) + g) * ROWS + r
    lse = tl.load(LSE + row, mask=smask, other=float("-inf"))
    m = tl.max(lse, 0)
    w = tl.math.exp2(lse - m)
    w = w / tl.sum(w, 0)
    op = tl.load(OP + row[:, None] * D + offs_d[None, :], mask=smask[:, None], other=0.0)
    o = tl.sum(op.to(tl.float32) * w[:, None], 0)
    t = r // GROUP
    h = g * GROUP + r % GROUP
    if HAS_GATE:
        o = o * tl.sigmoid(tl.load(G + t * sg_t + h * sg_h + offs_d).to(tl.float32))
    tl.store(O + t * so_t + h * so_h + offs_d, o.to(O.dtype.element_ty))


def decode_cfg(T):
    """Tuned at n=12000 (graph replay). PARTIAL: dtype of the per-split partial outputs (bf16 halves their traffic)."""
    rows = T * GROUP
    if rows <= 16:
        return dict(NSPLIT=64, BLOCK_N=32, BLOCK_M=16, PARTIAL=torch.bfloat16, num_warps=4, num_stages=2)
    if rows <= 64:
        return dict(NSPLIT=64, BLOCK_N=32, BLOCK_M=64, PARTIAL=torch.bfloat16, num_warps=4, num_stages=2)
    return dict(NSPLIT=32, BLOCK_N=64, BLOCK_M=64, PARTIAL=torch.bfloat16, num_warps=4, num_stages=2)

_workspace = {}


def decode_workspace(device, nsplit, rows, dtype):
    key = (device, nsplit, rows, dtype)
    if key not in _workspace:
        _workspace[key] = (torch.zeros(nsplit * KV_HEADS * rows, HEAD_DIM, dtype=dtype, device=device),
                           torch.zeros(nsplit * KV_HEADS * rows, dtype=torch.float32, device=device))
    return _workspace[key]


def attn_decode(q, k_cache, v_cache, n, gate=None, out=None, cfg=None):
    """q (T, 24, 256) bf16, T <= 16, at positions n-T..n-1; n: int32 device tensor (shape (1,) or ()).
    Returns o (T, 24, 256) bf16. Graph-safe: no host sync; the launch depends only on T."""
    T = q.shape[0]
    assert 1 <= T <= 16 and q.shape[1:] == (HEADS, HEAD_DIM) and q.stride(2) == 1
    assert k_cache.stride() == v_cache.stride() and k_cache.stride(2) == 1
    cfg = dict(decode_cfg(T), **(cfg or {}))
    NSPLIT, BLOCK_N = cfg.pop("NSPLIT"), cfg.pop("BLOCK_N")
    assert NSPLIT % 8 == 0
    rows = max(16, triton.next_power_of_2(T * GROUP))
    BM = min(cfg.pop("BLOCK_M"), rows)
    NRB = triton.cdiv(T * GROUP, BM)
    op, lse = decode_workspace(q.device, NSPLIT, rows, cfg.pop("PARTIAL"))
    o = torch.empty(T, HEADS, HEAD_DIM, dtype=q.dtype, device=q.device) if out is None else out
    gt = gate if gate is not None else o
    _decode_split_kernel[(NSPLIT * NRB, KV_HEADS)](
        q, k_cache, v_cache, n, op, lse, T,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), SCALE * LOG2E,
        GROUP=GROUP, D=HEAD_DIM, BLOCK_M=BM, BLOCK_N=BLOCK_N, NSPLIT=NSPLIT, NRB=NRB, ROWS=rows, **cfg)
    _decode_merge_kernel[(T * GROUP, KV_HEADS)](
        op, lse, o, gt, o.stride(0), o.stride(1), gt.stride(0), gt.stride(1),
        GROUP=GROUP, D=HEAD_DIM, ROWS=rows, NSPLIT=NSPLIT, NSPLIT_P2=triton.next_power_of_2(NSPLIT),
        HAS_GATE=gate is not None, num_warps=4)
    return o
