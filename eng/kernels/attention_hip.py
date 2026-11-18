"""HIP decode/verify attention (eng/csrc/attn.hip), bound through eng.hip's libengk.so. Same contract as the
Triton attention.attn_decode; faster than it for T >= 12, equal at T = 8, slower for T <= 4.
attn_decode_best picks the faster one by T."""
import ctypes

import torch

from .. import hip as H
from .attention import HEADS, HEAD_DIM, KV_HEADS, GROUP, attn_decode

P, I = ctypes.c_void_p, ctypes.c_int
H.lib.attn_decode.argtypes = [I, P, I, I, P, P, P, I, I, I, P, P, I, P, I, I, P, I, I, I, P]
ROWS_WS = 16 * GROUP          # workspace rows per (split, KV head): 6 * max T
_ws = {}


def _workspace(device, nsplit, part_bf16):
    key = (device, nsplit, part_bf16)
    if key not in _ws:
        dt = torch.bfloat16 if part_bf16 else torch.float32
        _ws[key] = (torch.zeros(nsplit * KV_HEADS * ROWS_WS, HEAD_DIM, dtype=dt, device=device),
                    torch.zeros(nsplit * KV_HEADS * ROWS_WS, dtype=torch.float32, device=device))
    return _ws[key]


def attn_decode_hip(q, k_cache, v_cache, n, gate=None, out=None, n_add=0, nsplit=64, part_bf16=True, stages=3):
    """q (T, 24, 256) bf16, 1 <= T <= 16, at positions n-T..n-1 where n = n[0] + n_add (n: int32 device tensor;
    pass the start position with n_add=T). k_cache, v_cache (L, 4, 256) contiguous bf16. gate (T, 24, 256)
    view: o *= sigmoid(gate). out: (T, 24, 256) view to write. Graph-safe (no host sync); the workspace is
    allocated on the first call, so call once before capture."""
    T = q.shape[0]
    assert 1 <= T <= 16 and q.shape[1:] == (HEADS, HEAD_DIM) and q.stride(2) == 1 and q.dtype == torch.bfloat16
    assert k_cache.stride() == v_cache.stride() == (KV_HEADS * HEAD_DIM, HEAD_DIM, 1)
    assert n.dtype == torch.int32
    o = torch.empty(T, HEADS, HEAD_DIM, dtype=q.dtype, device=q.device) if out is None else out
    assert o.stride(2) == 1
    op, lse = _workspace(q.device, nsplit, part_bf16)
    g = gate if gate is not None else None
    if g is not None:
        assert g.stride(2) == 1
    rc = H.lib.attn_decode(T, H.p(q), q.stride(0), q.stride(1), H.p(k_cache), H.p(v_cache), H.p(n), n_add, nsplit,
                           int(part_bf16), H.p(op), H.p(lse), ROWS_WS, H.p(o), o.stride(0), o.stride(1),
                           P(0) if g is None else H.p(g), 0 if g is None else g.stride(0),
                           0 if g is None else g.stride(1), stages, H.stream())
    if rc:
        raise RuntimeError(f"attn_decode: hip error {rc}")
    return o


def attn_decode_best(q, k_cache, v_cache, n, gate=None, out=None):
    """The faster of the HIP and Triton kernels for this T (measured at n = 4000..16000)."""
    if q.shape[0] >= 8:
        return attn_decode_hip(q, k_cache, v_cache, n, gate=gate, out=out)
    return attn_decode(q, k_cache, v_cache, n, gate=gate, out=out)
