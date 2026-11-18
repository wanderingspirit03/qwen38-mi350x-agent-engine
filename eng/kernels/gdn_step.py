"""Gated DeltaNet decode/verify kernels (Triton): T <= 16 new positions of one sequence.

gdn_verify: conv1d (4 taps) over [conv history; T new rows] + silu, l2norm of q and k, the delta-rule recurrence
from the committed state (not written back), the gated RMSNorm; writes the T output rows.
gdn_commit: once the number of accepted rows n is known, advances every GDN layer's state and conv history by
the first n rows (recomputing them), for all layers in one launch.
"""
import torch
import triton
import triton.language as tl

QK, VD, C = 2048, 6144, 10240
PROJ = C + VD + 96        # qkv | z | b | a


@triton.jit
def _row(proj_ptr, conv_ptr, r: tl.constexpr, ch, PROJ_STRIDE: tl.constexpr):
    # row r of [conv history (3 rows); proj rows]
    if r < 3:
        return tl.load(conv_ptr + r * 10240 + ch).to(tl.float32)
    else:
        return tl.load(proj_ptr + (r - 3) * PROJ_STRIDE + ch).to(tl.float32)


@triton.jit
def _row_dyn(proj_ptr, conv_ptr, r, ch, PROJ_STRIDE: tl.constexpr):
    hist = tl.load(conv_ptr + r * 10240 + ch, mask=(ch >= 0) & (r < 3), other=0.0).to(tl.float32)
    new = tl.load(proj_ptr + (r - 3) * PROJ_STRIDE + ch, mask=(ch >= 0) & (r >= 3), other=0.0).to(tl.float32)
    return tl.where(r < 3, hist, new)


@triton.jit
def _conv(proj_ptr, conv_ptr, cw_ptr, t, ch, PROJ_STRIDE: tl.constexpr):
    acc = _row_dyn(proj_ptr, conv_ptr, t, ch, PROJ_STRIDE) * tl.load(cw_ptr + ch * 4 + 0).to(tl.float32)
    acc += _row_dyn(proj_ptr, conv_ptr, t + 1, ch, PROJ_STRIDE) * tl.load(cw_ptr + ch * 4 + 1).to(tl.float32)
    acc += _row_dyn(proj_ptr, conv_ptr, t + 2, ch, PROJ_STRIDE) * tl.load(cw_ptr + ch * 4 + 2).to(tl.float32)
    acc += _row_dyn(proj_ptr, conv_ptr, t + 3, ch, PROJ_STRIDE) * tl.load(cw_ptr + ch * 4 + 3).to(tl.float32)
    return acc * tl.sigmoid(acc)


@triton.jit
def _step(proj_ptr, conv_ptr, cw_ptr, aneg_ptr, dtb_ptr, S, t, h, kh, offs, PROJ_STRIDE: tl.constexpr):
    q = _conv(proj_ptr, conv_ptr, cw_ptr, t, kh * 128 + offs, PROJ_STRIDE)
    k = _conv(proj_ptr, conv_ptr, cw_ptr, t, 2048 + kh * 128 + offs, PROJ_STRIDE)
    v = _conv(proj_ptr, conv_ptr, cw_ptr, t, 4096 + h * 128 + offs, PROJ_STRIDE)
    # the reference runs the conv in bf16 and normalizes from there
    q = q.to(tl.bfloat16).to(tl.float32)
    k = k.to(tl.bfloat16).to(tl.float32)
    v = v.to(tl.bfloat16).to(tl.float32)
    q = q * tl.rsqrt(tl.sum(q * q, 0) + 1e-6) * 0.08838834764831845
    k = k * tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
    b = tl.load(proj_ptr + t * PROJ_STRIDE + 16384 + h).to(tl.float32)
    a = tl.load(proj_ptr + t * PROJ_STRIDE + 16432 + h).to(tl.float32)
    x = a + tl.load(dtb_ptr + h)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = tl.load(aneg_ptr + h) * sp
    beta = tl.sigmoid(b)
    S = S * tl.exp(g)
    kv = tl.sum(S * k[:, None], 0)
    delta = (v - kv) * beta
    S = S + k[:, None] * delta[None, :]
    o = tl.sum(S * q[:, None], 0)
    return S, o


@triton.jit
def _verify_kernel(proj_ptr, conv_ptr, cw_ptr, aneg_ptr, dtb_ptr, gw_ptr, ssm_ptr, out_ptr,
                   T: tl.constexpr, PROJ_STRIDE: tl.constexpr):
    h = tl.program_id(0)
    kh = h // 3
    offs = tl.arange(0, 128)
    S = tl.load(ssm_ptr + h * 16384 + offs[:, None] * 128 + offs[None, :])
    gw = tl.load(gw_ptr + offs).to(tl.float32)
    for t in tl.static_range(T):
        S, o = _step(proj_ptr, conv_ptr, cw_ptr, aneg_ptr, dtb_ptr, S, t, h, kh, offs, PROJ_STRIDE)
        z = tl.load(proj_ptr + t * PROJ_STRIDE + 10240 + h * 128 + offs).to(tl.float32)
        ob = o.to(tl.bfloat16).to(tl.float32)
        y = ob * tl.rsqrt(tl.sum(ob * ob, 0) / 128.0 + 1e-6)
        y = (y.to(tl.bfloat16) * gw.to(tl.bfloat16)).to(tl.float32) * (z * tl.sigmoid(z))
        tl.store(out_ptr + t * 6144 + h * 128 + offs, y.to(tl.bfloat16))


@triton.jit
def _commit_kernel(proj_ptr, conv_ptr, cw_ptr, aneg_ptr, dtb_ptr, ssm_ptr, n_ptr,
                   PROJ_STRIDE: tl.constexpr, LAYER_PROJ: tl.constexpr):
    h = tl.program_id(0)
    layer = tl.program_id(1)
    kh = h // 3
    offs = tl.arange(0, 128)
    n = tl.load(n_ptr)
    proj_ptr += layer * LAYER_PROJ
    conv_ptr += layer * 3 * 10240
    cw_ptr += layer * 10240 * 4
    aneg_ptr += layer * 48
    dtb_ptr += layer * 48
    sp = ssm_ptr + layer * 48 * 16384 + h * 16384 + offs[:, None] * 128 + offs[None, :]
    S = tl.load(sp)
    for t in range(0, n):
        S, o = _step(proj_ptr, conv_ptr, cw_ptr, aneg_ptr, dtb_ptr, S, t, h, kh, offs, PROJ_STRIDE)
    tl.store(sp, S)


@triton.jit
def _conv_commit_kernel(proj_ptr, conv_ptr, n_ptr, PROJ_STRIDE: tl.constexpr, LAYER_PROJ: tl.constexpr,
                        BLOCK: tl.constexpr):
    layer = tl.program_id(1)
    ch = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = tl.load(n_ptr)
    proj_ptr += layer * LAYER_PROJ
    conv_ptr += layer * 3 * 10240
    r0 = _row_dyn(proj_ptr, conv_ptr, n, ch, PROJ_STRIDE)
    r1 = _row_dyn(proj_ptr, conv_ptr, n + 1, ch, PROJ_STRIDE)
    r2 = _row_dyn(proj_ptr, conv_ptr, n + 2, ch, PROJ_STRIDE)
    tl.debug_barrier()
    tl.store(conv_ptr + ch, r0.to(tl.bfloat16))
    tl.store(conv_ptr + 10240 + ch, r1.to(tl.bfloat16))
    tl.store(conv_ptr + 20480 + ch, r2.to(tl.bfloat16))


def gdn_verify(proj, conv, cw, aneg, dtb, gw, ssm, out):
    """proj (T, PROJ) bf16 rows of this layer's input projection; conv (3, 10240) bf16; cw (10240, 4);
    aneg, dtb (48,) fp32; gw (128,); ssm (48, 128, 128) fp32 (read only); out (T, 6144) bf16."""
    T = proj.shape[0]
    _verify_kernel[(48,)](proj, conv, cw, aneg, dtb, gw, ssm, out, T=T, PROJ_STRIDE=proj.stride(0),
                          num_warps=8)


def gdn_commit(proj_all, conv_all, cw_all, aneg_all, dtb_all, ssm_all, n_dev):
    """Advance all layers by the first n_dev[0] rows. proj_all (L, T, PROJ); conv_all (L, 3, 10240);
    cw_all (L, 10240, 4); aneg_all, dtb_all (L, 48); ssm_all (L, 48, 128, 128); n_dev int32 (1,) on device."""
    L = proj_all.shape[0]
    _commit_kernel[(48, L)](proj_all, conv_all, cw_all, aneg_all, dtb_all, ssm_all, n_dev,
                            PROJ_STRIDE=proj_all.stride(1), LAYER_PROJ=proj_all.stride(0), num_warps=8)
    _conv_commit_kernel[(10240 // 1024, L)](proj_all, conv_all, n_dev, PROJ_STRIDE=proj_all.stride(1),
                                            LAYER_PROJ=proj_all.stride(0), BLOCK=1024, num_warps=4)
