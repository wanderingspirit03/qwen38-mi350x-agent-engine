"""W4A8 decode GEMM (M = 1..16 rows): y (M, N) = x (M, K) @ W^T with 4-bit weights and int8 activations.

Weights: asymmetric int4 per (row, group of 128) with an fp16 scale and a zero point (the same quantization as
eng.kernels.w4: clipping search), packed for v_mfma_i32_16x16x64_i8. Activations: symmetric int8 per (row, group
of 128), made by quant_x() (one small kernel) from the bf16 x.

API:
    p = quantize(w, norm_weight=None, swiglu=False)      as w4.quantize (G = 128 only), but the norm weight is
        not folded into W: quant_x applies it to x (folding it into int4 W doubles the error in some layers)
    xa = quant_x(x, xa=None)                             x (M, K) bf16 -> activation dict (int8 + scales)
    y = gemm(x_or_xa, p, out=None, residual=None)        bf16 x: quant_x then the GEMM (two kernels)
HIP-graph safe when out (and xa) are given.
"""
import ctypes
import hashlib
import os
import subprocess

import torch

from ._w4a8_src import HIP_SRC
from . import w4

_lib = None


def _build(prefix=None):
    global _lib
    if _lib is not None and prefix is None:
        return _lib
    src_text = (prefix or "") + HIP_SRC
    h = hashlib.sha1(src_text.encode()).hexdigest()[:16]
    d = os.environ.get("ENG_KERNEL_CACHE", "/tmp/eng-kernels")
    os.makedirs(d, exist_ok=True)
    so = os.path.join(d, f"w4a8_{h}.so")
    if not os.path.exists(so):
        src = os.path.join(d, f"w4a8_{h}.hip")
        with open(src, "w") as f:
            f.write(src_text)
        tmp = so + f".{os.getpid()}.tmp"
        subprocess.run(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx950", "-shared", "-fPIC",
                        "-o", tmp, src], check=True)
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.w4a8_gemm.restype = ctypes.c_int
    lib.w4a8_gemm.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 7 + [ctypes.c_float, ctypes.c_void_p]
    lib.w4a8_quant_x.restype = ctypes.c_int
    lib.w4a8_quant_x.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    if prefix is None:
        _lib = lib
    return lib


def _ngp(K):
    return -(-(K // 128) // 32) * 32


def _pack(u, s, z):
    """u (n, NG, 128) int codes 0..15, s, z (n, NG) -> qw (n/16, NG, 256) int32, msc (n/16, NG, 16) fp16,
    mzp (n/16, 16, NGp) bf16."""
    n, NG, _ = u.shape
    v = (u ^ 8).to(torch.uint8).reshape(n // 16, 16, NG, 2, 4, 4, 4)       # (tile, r, g, half, lg, d, b)
    byte = v[:, :, :, 0] | (v[:, :, :, 1] << 4)                            # (tile, r, g, lg, d, b)
    qw = byte.permute(0, 2, 3, 1, 4, 5).contiguous().view(torch.int32).reshape(n // 16, NG, 256)
    sh = s.half()
    msc = sh.reshape(n // 16, 16, NG).transpose(1, 2).contiguous()
    zp = torch.zeros(n, _ngp(NG * 128), dtype=torch.bfloat16, device=u.device)
    zp[:, :NG] = (-sh.float() * (z - 8)).bfloat16()
    return qw, msc, zp.reshape(n // 16, 16, -1)


@torch.no_grad()
def quantize(w, norm_weight=None, swiglu=False, clip_search=True, rows_per_chunk=8192):
    N, K = w.shape
    assert N % 16 == 0 and K % 128 == 0
    if swiglu:
        assert N % 32 == 0
        w = w.reshape(2, N // 32, 16, K).transpose(0, 1).reshape(N, K)
    qs, ss, zs = [], [], []
    for r0 in range(0, N, rows_per_chunk):
        wc = w[r0:r0 + rows_per_chunk].float()
        u, s, z = w4.rtn_search(wc.reshape(-1, K // 128, 128), clip_search)
        q, sc, zp = _pack(u, s, z)
        qs.append(q)
        ss.append(sc)
        zs.append(zp)
    return {"qw": torch.cat(qs).contiguous(), "msc": torch.cat(ss).contiguous(), "mzp": torch.cat(zs).contiguous(),
            "N": N, "K": K, "rms": norm_weight is not None, "swiglu": swiglu,
            "norm": None if norm_weight is None else norm_weight.bfloat16().contiguous()}


def dequantize(p):
    """(N, K) fp32 weights as the kernel uses them (rows interleaved if swiglu)."""
    N, K = p["N"], p["K"]
    NG = K // 128
    b = p["qw"].reshape(N // 16, NG, 4, 16, 4).contiguous().view(torch.uint8).reshape(N // 16, NG, 4, 16, 4, 4)
    lo, hi = (b & 15).int() ^ 8, (b >> 4).int() ^ 8                        # (tile, g, lg, r, d, b)
    u = torch.stack([lo, hi], 2).permute(0, 4, 1, 2, 3, 5, 6).reshape(N, NG, 128).float()
    s = p["msc"].float().transpose(1, 2).reshape(N, NG)
    zp = p["mzp"].float().reshape(N, -1)[:, :NG]                           # -s (z - 8), as the kernel uses it
    return ((u - 8) * s[..., None] + zp[..., None]).reshape(N, K)


def new_xa(M, K, device="cuda"):
    NG = K // 128
    return {"xq": torch.empty(M, K, dtype=torch.int8, device=device),
            "xg": torch.empty(M, NG, 2, dtype=torch.float32, device=device),
            "xzp": torch.empty(M, _ngp(K), dtype=torch.bfloat16, device=device), "M": M, "K": K}


def quant_x(x, xa=None, norm_weight=None):
    """x (M, K) bf16 -> int8 per (row, group of 128) of x (1 + norm_weight) (or of x), plus the sums the GEMM needs.
    With a norm weight the GEMM (p["rms"]) applies 1/rms(x) itself."""
    lib = _build()
    M, K = x.shape
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and K % 128 == 0
    if xa is None:
        xa = new_xa(M, K, x.device)
    assert xa["M"] == M and xa["K"] == K
    lib.w4a8_quant_x(x.data_ptr(), norm_weight.data_ptr() if norm_weight is not None else None, xa["xq"].data_ptr(), xa["xg"].data_ptr(), xa["xzp"].data_ptr(), M, K,
                     torch.cuda.current_stream().cuda_stream)
    return xa


def dequant_x(xa):
    M, K = xa["M"], xa["K"]
    sx = xa["xg"][..., 0] * 16
    return (xa["xq"].float().reshape(M, K // 128, 128) * sx[..., None]).reshape(M, K)


def reference(x, p, residual=None):
    """fp32: int8 x (as quant_x makes it) times the dequantized weights, with the fusions."""
    xf = x.float()
    y = dequant_x(quant_x(x, norm_weight=p["norm"])) @ dequantize(p).t()
    if p["rms"]:
        y = y * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    if p["swiglu"]:
        y = y.reshape(y.shape[0], -1, 2, 16)
        g, u = y[:, :, 0].bfloat16().float(), y[:, :, 1].bfloat16().float()
        return (torch.nn.functional.silu(g) * u).reshape(y.shape[0], -1)
    return y if residual is None else y + residual.float()


def nbytes(p):
    return p["qw"].numel() * 4 + p["msc"].numel() * 2 + p["mzp"].numel() * 2


EPS = 1e-6
# (row tiles of 16 per wave, waves per workgroup) per (N, K) as a list of (largest M, config); tuned on MI350X
CONFIGS = {
    (34816, 5120): [(4, (2, 2)), (16, (4, 2))],      # MLP gate+up
    (5120, 17408): [(16, (2, 8))],                   # MLP down
    (16480, 5120): [(12, (2, 4)), (16, (2, 2))],     # GDN in_proj
    (5120, 6144): [(16, (2, 8))],                    # GDN / attention out
    (14336, 5120): [(4, (2, 4)), (16, (4, 8))],      # attention q+gate,k,v
    (5120, 10240): [(16, (2, 8))],                   # MTP fc
    (32768, 5120): [(12, (2, 2)), (16, (4, 4))],     # draft head
}
DEFAULT = [(16, (2, 4))]


def config_for(N, K, M):
    for m, c in CONFIGS.get((N, K), DEFAULT):
        if M <= m:
            return c


def gemm(x, p, out=None, residual=None, cfg=None, xa=None):
    """x: (M, K) bf16 (quantized here into xa) or an activation dict from quant_x."""
    lib = _build()
    if isinstance(x, dict):
        xa = x
    else:
        xa = quant_x(x, xa, p["norm"])
    M, K = xa["M"], xa["K"]
    N = p["N"]
    assert K == p["K"] and 1 <= M <= 16
    No = N // 2 if p["swiglu"] else N
    assert residual is None or (not p["swiglu"] and residual.shape == (M, N) and residual.is_contiguous())
    if out is None:
        out = torch.empty(M, No, dtype=torch.bfloat16, device=xa["xq"].device)
    nt, waves = cfg or config_for(N, K, M)
    assert not (p["swiglu"] and nt % 2) and N % (16 * nt) == 0
    assert nt * (K // 128) * 32 <= 65536
    rc = lib.w4a8_gemm(xa["xq"].data_ptr(), xa["xg"].data_ptr(), xa["xzp"].data_ptr(), p["qw"].data_ptr(),
                       p["msc"].data_ptr(), p["mzp"].data_ptr(), out.data_ptr(),
                       residual.data_ptr() if residual is not None else None, M, N, K, nt, waves,
                       int(p["rms"]), int(p["swiglu"]), EPS, torch.cuda.current_stream().cuda_stream)
    assert rc == 0, "no kernel instance for this configuration"
    return out
