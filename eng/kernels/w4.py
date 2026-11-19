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
    p = quantize8(w, norm_weight=None): 8-bit weights (int8, groups of 128) for the same gemm().
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
    wpe = os.environ.get("ENG_GV_WPE")     # experiment: waves per EU of the M = 1 GEMV (0: no attribute)
    if wpe is not None:
        prefix = (prefix or "") + f"#define GV_WPE {wpe}\n"
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
    lib.w4_gemm.argtypes = ([ctypes.c_void_p] * 7 + [ctypes.c_int] * 9 + [ctypes.c_float] + [ctypes.c_void_p] * 2)
    lib.rd_bench.restype = ctypes.c_int
    lib.rd_bench.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.w4_gemv_mode.restype = ctypes.c_int
    lib.w4_gemv_mode.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int] * 8 + [ctypes.c_float] + [ctypes.c_void_p] * 2)
    lib.w4_gemv.restype = ctypes.c_int
    lib.w4_gemv.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int] * 9 + [ctypes.c_float] + [ctypes.c_void_p] * 4)
    if prefix is None or wpe is not None:
        _lib = lib
    return lib


_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]   # nibble p of a dword holds element _ORDER[p] of the lane's 8 k values


def _pack(u, s, z):
    """u (n, K/G, G) int32 codes 0..15, s (n, K/G) scales, z (n, K/G) zero points 0..15 -> (qw, meta) blocks."""
    n, NG, G = u.shape
    K = NG * G
    # (tile, r, kb, s, lg, e) -> (tile, kb, lg, r, s, e)
    u = u.reshape(n // 16, 16, K // 128, 4, 4, 8).permute(0, 2, 4, 1, 3, 5)
    u = u[..., _ORDER]
    sh = torch.arange(0, 32, 4, device=u.device, dtype=torch.int32)
    packed = (u << sh).sum(-1, dtype=torch.int64)
    packed = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)
    sc = s.half().reshape(n // 16, 16, NG).transpose(1, 2).contiguous().view(torch.uint8)  # (tiles, NG, 32)
    zc = (z + 128).to(torch.uint8).reshape(n // 16, 16, NG).transpose(1, 2)               # (tiles, NG, 16)
    return packed.reshape(n // 16, K // 128, 256), torch.cat([sc, zc], -1)


def rtn_search(wc, clip_search=True):
    """wc (n, NG, G) fp32 -> codes u (n, NG, G) int32 in 0..15, scales s and zero points z (n, NG) fp32 (s is
    fp16-representable): asymmetric round-to-nearest, clipping ratio per group chosen by least squares."""
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
    return u, s, z


@torch.no_grad()
def quantize(w, group=128, norm_weight=None, swiglu=False, clip_search=True, rows_per_chunk=8192, H=None):
    """w: (N, K) CUDA tensor. Returns the packed weight (dict). See the module docstring.
    H: X^T X of the GEMM's input (x / rms(x) if norm_weight is given): GPTQ (eng.gptq) instead of round-to-nearest."""
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
        if H is None:
            u, s, z = rtn_search(wc.reshape(-1, K // G, G), clip_search)
        else:
            from ..gptq import gptq_codes
            u, s, z = gptq_codes(wc, H, G)
        q, m = _pack(u, s, z)
        qs.append(q)
        ms.append(m)
    _workspace(w.device)
    return {"qw": torch.cat(qs).contiguous(), "meta": torch.cat(ms).contiguous(), "N": N, "K": K, "G": G,
            "rms": norm_weight is not None, "swiglu": swiglu}


@torch.no_grad()
def quantize8(w, norm_weight=None, rows_per_chunk=8192):
    """8-bit weights in the 4-bit format: symmetric int8 per group of 128, q8 = round(w / s) + 128 in 1..255 is split
    into its nibbles, q8 = 16 h + l, and w = 16 s (h - 8) + s (l - 0). Each 128-k block of W becomes two 4-bit
    blocks (h, then l) with their own scale and zero point, so the packed matrix is (N, 2K) and gemm() reads every
    x block twice ("xdup"). The RMSNorm fold is as in quantize(); sum(x^2) / 2K over the doubled x is the mean."""
    N, K = w.shape
    assert N % 16 == 0 and K % 128 == 0
    qs, ms = [], []
    for r0 in range(0, N, rows_per_chunk):
        wc = w[r0:r0 + rows_per_chunk].float()
        if norm_weight is not None:
            wc = wc * (1 + norm_weight.float())[None]
        n = wc.shape[0]
        wc = wc.reshape(n, K // 128, 128)
        s = (wc.abs().amax(-1) / 127).clamp(min=1e-8).half().float()
        q8 = (wc / s[..., None]).round().clamp_(-127, 127).to(torch.int32) + 128
        u = torch.stack([q8 >> 4, q8 & 15], 2).reshape(n, 2 * K // 128, 128)
        s2 = torch.stack([16 * s, s], 2).reshape(n, -1)
        z2 = torch.stack([torch.full_like(s, 8), torch.zeros_like(s)], 2).reshape(n, -1)
        q, m = _pack(u, s2, z2)
        qs.append(q)
        ms.append(m)
    _workspace(w.device)
    return {"qw": torch.cat(qs).contiguous(), "meta": torch.cat(ms).contiguous(), "N": N, "K": 2 * K, "G": 128,
            "rms": norm_weight is not None, "swiglu": False, "xdup": True}


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


def dequantize8(p):
    """(N, K) fp32 weights of a quantize8() result."""
    return dequantize(p).reshape(p["N"], -1, 2, 128).sum(2).reshape(p["N"], -1)


def reference(x, p, residual=None):
    """What gemm() computes, in fp32 PyTorch (from the dequantized weights)."""
    xf = x.float()
    y = xf @ (dequantize8(p) if p.get("xdup") else dequantize(p)).t()
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
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    if device not in _ws:
        _ws[device] = (torch.empty(WS_FLOATS, dtype=torch.float32, device=device),
                       torch.zeros(65536, dtype=torch.int32, device=device),
                       torch.zeros(16, dtype=torch.int64, device=device))     # argmax slots (see F_ARGMAX)
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
    (248320, 10240): [(4, (2, 8, 1)), (12, (4, 4, 1)), (16, (8, 1, 1))],              # LM head, 8-bit (quantize8)
    (32768, 10240): [(4, (2, 8, 1)), (12, (4, 4, 1)), (16, (8, 4, 1))],               # draft head, 8-bit
}
DEFAULT = [(4, (2, 4, 1)), (16, (2, 2, 1))]


def lds_bytes(nt, ksplit, K, G):
    return nt * -(-(K // 128) // ksplit) * (128 // G) * 48


# M = 1 GEMV (w4_gemv_kernel): (row tiles per workgroup, waves, weight ring depth) per (N, K)
GEMV = os.environ.get("ENG_GEMV", "1") == "1"
GV_CONFIGS = {    # eng/bench_gv.py, random weights (us: old w4 kernel -> this)
    (34816, 5120): (2, 4, 2),     # MTP gate+up      27.3 -> 24.6
    (5120, 17408): (2, 8, 2),     # MTP down         18.3 -> 15.1 (13.8 with i8)
    (14336, 5120): (2, 4, 1),     # MTP q+gate,k,v   12.7 -> 11.2
    (5120, 6144): (2, 8, 2),      # MTP o             8.8 -> 6.4
    (5120, 10240): (2, 8, 2),     # MTP fc           11.0 -> 8.2
    (33024, 5120): (2, 4, 2),     # draft head (32768 + EXTRA rows)  25.2 -> 23.6
}


GV_I8 = os.environ.get("ENG_GEMV_I8", "0") == "1"


def gemv_mode(x, p, out, cfg, mode):
    """Ablation of w4_gemv_kernel (bf16 x): mode 1 = loads + staging, no math; mode 2 = weight loads only."""
    lib = _build()
    N, K = p["N"], p["K"]
    nt, waves, d = cfg
    rc = lib.w4_gemv_mode(x.data_ptr(), p["qw"].data_ptr(), p["meta"].data_ptr(), out.data_ptr(), None, N, K, nt, waves,
                          d, int(p["swiglu"]), int(p["rms"]), mode, EPS, _workspace(x.device)[2].data_ptr(),
                          torch.cuda.current_stream().cuda_stream)
    assert rc == 0, f"w4_gemv_mode: no kernel for {(N, K, nt, waves, d, mode)}"


def gemv(x, p, out, residual=None, cfg=None, argmax=False, i8=None):
    """M = 1, G = 128: as gemm(); the streaming GEMV kernel. residual may be out (in place).
    i8: x quantized to int8 per 128-k block inside the kernel (int8 MFMAs)."""
    lib = _build()
    N, K = p["N"], p["K"]
    cfg = cfg or GV_CONFIGS[(N, K)]
    nt, waves, d = cfg[:3]
    sk = cfg[3] if len(cfg) > 3 else 1      # k splits (workgroups per row tiles)
    assert x.shape == (1, K) and p["G"] == 128 and not p.get("xdup") and not (p["swiglu"] and nt % 2)
    ws, cnt, am = _workspace(x.device)
    rc = lib.w4_gemv(x.data_ptr(), p["qw"].data_ptr(), p["meta"].data_ptr(), out.data_ptr(),
                     residual.data_ptr() if residual is not None else None, N, K, nt, waves, d, sk,
                     int(p["swiglu"]) | (4 if argmax else 0), int(p["rms"]), int(GV_I8 if i8 is None else i8), EPS,
                     am.data_ptr(), ws.data_ptr(), cnt.data_ptr(),
                     torch.cuda.current_stream().cuda_stream)
    assert rc == 0, f"w4_gemv: no kernel for {(N, K, cfg)} ({rc})"
    return out


def config_for(N, K, M):
    for m, c in CONFIGS.get((N, K), DEFAULT):
        if M <= m:
            return c


def argmax_slots(device):
    """The slots gemm(argmax=True) leaves its argmax in; eng.hip.am_decode / mtp_in read and zero them."""
    return _workspace(device)[2]


def gemm(x, p, out=None, residual=None, cfg=None, argmax=False):
    """x: (M, K) bf16 contiguous, M <= 16. Returns (M, N) bf16 = x @ W^T (+ residual). See module docstring.
    argmax: also leave the argmax of the bf16 output row in argmax_slots() (M = 1)."""
    lib = _build()
    M = x.shape[0]
    N, K = p["N"], p["K"]
    xdup = p.get("xdup", False)
    assert x.shape[1] * (2 if xdup else 1) == K and 1 <= M <= 16 and x.dtype == torch.bfloat16 and x.is_contiguous()
    No = N // 2 if p["swiglu"] else N
    assert residual is None or (not p["swiglu"] and residual.shape == (M, N) and residual.is_contiguous())
    if out is None:
        out = torch.empty(M, No, dtype=torch.bfloat16, device=x.device)
    if M == 1 and cfg is None and GEMV and (N, K) in GV_CONFIGS and p["G"] == 128 and not xdup:
        return gemv(x, p, out, residual, argmax=argmax)
    nt, waves, ksplit = cfg or config_for(N, K, M)
    assert not (p["swiglu"] and nt % 2), "swiglu needs an even number of row tiles per wave"
    assert N % (16 * nt) == 0 and N // (16 * nt) <= 65536 and waves in (1, 2, 4, 8, 16)
    ws, cnt, am = _workspace(x.device)
    assert not argmax or (M == 1 and ksplit == 1 and not p["swiglu"] and residual is None)
    assert ksplit == 1 or ksplit * (M * N + N // nt * 16) <= WS_FLOATS
    assert lds_bytes(nt, ksplit, K, p["G"]) <= 65536, "scales do not fit in LDS"
    rc = lib.w4_gemm(x.data_ptr(), p["qw"].data_ptr(), p["meta"].data_ptr(),
                     out.data_ptr(), residual.data_ptr() if residual is not None else None,
                     ws.data_ptr(), cnt.data_ptr(), M, N, K, p["G"], nt, waves, ksplit,
                     int(p["rms"]), int(p["swiglu"]) | (2 if xdup else 0) | (4 if argmax else 0), EPS, am.data_ptr(),
                     torch.cuda.current_stream().cuda_stream)
    assert rc == 0, "no kernel instance for this configuration"
    return out
