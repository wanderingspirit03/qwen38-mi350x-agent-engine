"""Weight-only 4-bit GEMM for decode (M = 1..16 rows): y (M, N) = x (M, K) @ W^T.

Format: asymmetric int4, group size G (128 or 64) along K, an fp16 scale and a uint8 (128 + zero point) per
(row, group): 4.19 bits per weight at G=128. The weights are pre-shuffled so that each lane of a wave loads
16 contiguous bytes that map directly onto bf16 MFMA operands (v_mfma_f32_16x16x32_bf16). Scales and zero points
are stored per (16-row tile, group) as one 48-byte record ("meta"); each workgroup copies its records into LDS.

API:
    p = quantize(w, group=128, norm_weight=None, swiglu=False)
        w (N, K) bf16 on the GPU -> dict of packed tensors.
        norm_weight (K,): the RMSNorm in front of this matrix, x_norm = x / rms(x) * (1 + norm_weight). It is
            folded into W, and gemm() then takes the un-normalized x and computes 1/rms(x) itself.
        swiglu: w is [gate; up] (2I, K); gemm() returns silu(x @ gate^T) * (x @ up^T), (M, I).
    y = gemm(x, p, out=None, residual=None)
        x (M, K) bf16 contiguous, M <= 16 -> (M, N) bf16 (+ residual; residual may be out itself: in place).
    w = dequantize(p)    (N, K) fp32 (norm weight folded in, rows interleaved if swiglu), for checks.
No host synchronization in gemm(): it can be captured in a HIP graph (call it once before capturing, so that the
kernels are compiled and loaded). The split-k workspace and counters are shared per device: do not run two
split-k gemm() calls concurrently on different streams.
"""
import ctypes
import hashlib
import os
import subprocess

import torch

from ._w4_src import HIP_SRC

_lib = None


def _build(prefix=None):
    """Compile (cached by source hash) and load the kernels. prefix: extra source lines put first (experiments)."""
    global _lib
    if _lib is not None and prefix is None:
        return _lib
    src_text = (prefix or "") + HIP_SRC
    h = hashlib.sha1(src_text.encode()).hexdigest()[:16]
    d = os.environ.get("ENG_KERNEL_CACHE", "/tmp/eng-kernels")
    os.makedirs(d, exist_ok=True)
    so = os.path.join(d, f"w4_{h}.so")
    if not os.path.exists(so):
        src = os.path.join(d, f"w4_{h}.hip")
        with open(src, "w") as f:
            f.write(src_text)
        tmp = so + f".{os.getpid()}.tmp"
        subprocess.run(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx950", "-shared", "-fPIC",
                        "-o", tmp, src], check=True)
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.w4_gemm.restype = ctypes.c_int
    lib.w4_gemm.argtypes = ([ctypes.c_void_p] * 7 + [ctypes.c_int] * 9 + [ctypes.c_float, ctypes.c_void_p])
    if prefix is None:
        _lib = lib
    return lib


_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]   # nibble p of a dword holds element _ORDER[p] of the lane's 8 k values


@torch.no_grad()
def quantize(w, group=128, norm_weight=None, swiglu=False, clip_search=True, rows_per_chunk=8192):
    """w: (N, K) CUDA tensor. Returns the packed weight (dict). See the module docstring."""
    N, K = w.shape
    assert N % 16 == 0 and K % 128 == 0 and group in (64, 128)
    G = group
    if swiglu:
        assert N % 32 == 0
        w = w.reshape(2, N // 32, 16, K).transpose(0, 1).reshape(N, K)   # 16-row tiles: gate, up, gate, ...
    qs, ms = [], []
    for r0 in range(0, N, rows_per_chunk):
        wc = w[r0:r0 + rows_per_chunk].float()
        if norm_weight is not None:
            wc = wc * (1 + norm_weight.float())[None]
        wc = wc.reshape(-1, K // G, G)
        lo = wc.amin(-1).clamp(max=0)
        hi = wc.amax(-1).clamp(min=0)
        best_err = best_s = best_z = None
        for a in (torch.linspace(1.0, 0.75, 11).tolist() if clip_search else [1.0]):
            s = ((hi - lo) * a / 15).clamp(min=1e-8).half().float()
            z = (-lo * a / s).round().clamp(0, 15)
            u = (wc / s[..., None]).round().add_(z[..., None]).clamp_(0, 15)
            err = ((u - z[..., None]).mul_(s[..., None]).sub_(wc)).square_().sum(-1)
            if best_err is None:
                best_err, best_s, best_z = err, s, z
            else:
                b = err < best_err
                best_err = torch.where(b, err, best_err)
                best_s = torch.where(b, s, best_s)
                best_z = torch.where(b, z, best_z)
        s, z = best_s, best_z
        u = (wc / s[..., None]).round().add_(z[..., None]).clamp_(0, 15).to(torch.int32)
        n = u.shape[0]
        # (tile, r, kb, s, lg, e) -> (tile, kb, lg, r, s, e)
        u = u.reshape(n // 16, 16, K // 128, 4, 4, 8).permute(0, 2, 4, 1, 3, 5)
        u = u[..., _ORDER]
        sh = torch.arange(0, 32, 4, device=w.device, dtype=torch.int32)
        packed = (u << sh).sum(-1, dtype=torch.int64)
        packed = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)
        qs.append(packed.reshape(n // 16, K // 128, 256))
        sc = s.half().reshape(n // 16, 16, K // G).transpose(1, 2).contiguous().view(torch.uint8)  # (tiles, NG, 32)
        zc = (z + 128).to(torch.uint8).reshape(n // 16, 16, K // G).transpose(1, 2)               # (tiles, NG, 16)
        ms.append(torch.cat([sc, zc], -1))
    _workspace(w.device)
    return {"qw": torch.cat(qs).contiguous(), "meta": torch.cat(ms).contiguous(), "N": N, "K": K, "G": G,
            "rms": norm_weight is not None, "swiglu": swiglu}


def dequantize(p):
    N, K, G = p["N"], p["K"], p["G"]
    q = p["qw"].reshape(N // 16, K // 128, 4, 16, 4, 1)
    sh = torch.arange(0, 32, 4, device=q.device, dtype=torch.int32)
    u = (q >> sh) & 15                                    # (tile, kb, lg, r, s, nibble)
    u = u[..., [_ORDER.index(e) for e in range(8)]]       # (tile, kb, lg, r, s, e)
    u = u.permute(0, 3, 1, 4, 2, 5).reshape(N, K).float()
    m = p["meta"]
    s = m[..., :32].contiguous().view(torch.float16).transpose(1, 2).reshape(N, K // G).float()
    z = m[..., 32:].transpose(1, 2).reshape(N, K // G).float() - 128
    return ((u.reshape(N, K // G, G) - z[..., None]) * s[..., None]).reshape(N, K)


def reference(x, p, residual=None):
    """What gemm() computes, in fp32 PyTorch (from the dequantized weights)."""
    xf = x.float()
    y = xf @ dequantize(p).t()
    if p["rms"]:
        y = y * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    if p["swiglu"]:
        y = y.reshape(y.shape[0], -1, 2, 16)
        g, u = y[:, :, 0].bfloat16().float(), y[:, :, 1].bfloat16().float()
        return (torch.nn.functional.silu(g) * u).reshape(y.shape[0], -1)
    return y if residual is None else y + residual.float()


def nbytes(p):
    return p["qw"].numel() * 4 + p["meta"].numel()


EPS = 1e-6
WS_FLOATS = 8 << 20
_ws = {}


def _workspace(device):
    """Split-k partial sums (fp32) and per-row-tile arrival counters (reset by the kernel after use)."""
    device = torch.device(device)
    if device not in _ws:
        _ws[device] = (torch.empty(WS_FLOATS, dtype=torch.float32, device=device),
                       torch.zeros(65536, dtype=torch.int32, device=device))
    return _ws[device]


# Configurations: (row tiles of 16 per wave, waves per workgroup, k splits), tuned on MI350X per (N, K) as a
# list of (largest M, config). Other shapes get DEFAULT.
CONFIGS = {
    (34816, 5120): [(2, (2, 4, 1)), (4, (2, 2, 1)), (12, (4, 4, 1)), (16, (4, 2, 1))],  # MLP gate+up
    (5120, 17408): [(16, (2, 8, 1))],                                                 # MLP down
    (16480, 5120): [(4, (2, 4, 1)), (16, (2, 2, 1))],                                 # GDN in_proj
    (5120, 6144): [(12, (2, 8, 1)), (16, (2, 4, 1))],                                 # GDN / attention out
    (14336, 5120): [(16, (4, 4, 1))],                                                 # attention q+gate,k,v
    (248320, 5120): [(4, (2, 1, 1)), (12, (4, 1, 1)), (16, (8, 1, 1))],               # LM head
    (5120, 10240): [(16, (2, 8, 1))],                                                 # MTP fc
    (32768, 5120): [(4, (2, 8, 1)), (16, (4, 4, 1))],                                 # draft head
}
DEFAULT = [(4, (2, 4, 1)), (16, (2, 2, 1))]


def lds_bytes(nt, ksplit, K, G):
    return nt * -(-(K // 128) // ksplit) * (128 // G) * 48


def config_for(N, K, M):
    for m, c in CONFIGS.get((N, K), DEFAULT):
        if M <= m:
            return c


def gemm(x, p, out=None, residual=None, cfg=None):
    """x: (M, K) bf16 contiguous, M <= 16. Returns (M, N) bf16 = x @ W^T (+ residual). See module docstring."""
    lib = _build()
    M, K = x.shape
    N = p["N"]
    assert K == p["K"] and 1 <= M <= 16 and x.dtype == torch.bfloat16 and x.is_contiguous()
    No = N // 2 if p["swiglu"] else N
    assert residual is None or (not p["swiglu"] and residual.shape == (M, N) and residual.is_contiguous())
    if out is None:
        out = torch.empty(M, No, dtype=torch.bfloat16, device=x.device)
    nt, waves, ksplit = cfg or config_for(N, K, M)
    assert not (p["swiglu"] and nt % 2), "swiglu needs an even number of row tiles per wave"
    assert N % (16 * nt) == 0 and N // (16 * nt) <= 65536 and waves in (1, 2, 4, 8, 16)
    ws, cnt = _workspace(x.device)
    assert ksplit == 1 or ksplit * (M * N + N // nt * 16) <= WS_FLOATS
    assert lds_bytes(nt, ksplit, K, p["G"]) <= 65536, "scales do not fit in LDS"
    rc = lib.w4_gemm(x.data_ptr(), p["qw"].data_ptr(), p["meta"].data_ptr(),
                     out.data_ptr(), residual.data_ptr() if residual is not None else None,
                     ws.data_ptr(), cnt.data_ptr(), M, N, K, p["G"], nt, waves, ksplit,
                     int(p["rms"]), int(p["swiglu"]), EPS, torch.cuda.current_stream().cuda_stream)
    assert rc == 0, "no kernel instance for this configuration"
    return out
