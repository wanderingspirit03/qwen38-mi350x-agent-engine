"""ctypes bindings of the engine's HIP kernels (eng/csrc/*.hip, compiled into libengk.so by the Dockerfile)."""
import ctypes
import os

import torch

_LIB = os.environ.get("ENG_HIPLIB", os.path.join(os.path.dirname(__file__), "libengk.so"))
lib = ctypes.CDLL(os.path.abspath(_LIB))
P = ctypes.c_void_p
I = ctypes.c_int
L = ctypes.c_long


def p(t):
    return P(t.data_ptr())


def stream():
    return P(torch.cuda.current_stream().cuda_stream)


lib.gdn_prep.argtypes = [I, P, I, P, P, P, P, P, P, P, P]
lib.gdn_verify.argtypes = [I, P, P, P, P, P, P, I, P, P, P]
lib.gdn_commit.argtypes = [I, I, P, P, P, P, P, P, I, L, P, P]
lib.gdn_verify_q.argtypes = [I, P, P, P, P, P, P, I, P, P, P, P, I, P]


def _check(rc, name):
    if rc:
        raise RuntimeError(f"{name}: hip error {rc}")


def gdn_prep(proj, hist, cw, aneg, dtb, qk, v, gb):
    """proj (T, PROJ) bf16 (row stride may be larger), hist (3, 10240) bf16, cw (10240, 4) bf16, aneg/dtb (48,)
    fp32 -> qk (T, 32, 128), v (T, 48, 128), gb (T, 2, 48) fp32 (all written)."""
    T = proj.shape[0]
    _check(lib.gdn_prep(T, p(proj), proj.stride(0), p(hist), p(cw), p(aneg), p(dtb), p(qk), p(v), p(gb), stream()),
           "gdn_prep")


def gdn_verify(qk, v, gb, state, o, proj, gw, out):
    """state (48, 128 v, 128 k) fp32 (read only); o (>= T, 48, 128) fp32 scratch; out (T, 6144) bf16."""
    T = proj.shape[0]
    _check(lib.gdn_verify(T, p(qk), p(v), p(gb), p(state), p(o), p(proj), proj.stride(0), p(gw), p(out), stream()),
           "gdn_verify")


def gdn_verify_q(qk, v, gb, state, o, proj, gw, xa):
    """gdn_verify with the output quantized into xa (an eng.kernels.w4a8 activation dict, M = T, K = 6144)."""
    T = proj.shape[0]
    assert xa["M"] == T and xa["K"] == 6144
    _check(lib.gdn_verify_q(T, p(qk), p(v), p(gb), p(state), p(o), p(proj), proj.stride(0), p(gw), p(xa["xq"]),
                            p(xa["xg"]), p(xa["xzp"]), xa["xzp"].shape[1], stream()), "gdn_verify_q")


def gdn_commit(n_dev, qk_all, v_all, gb_all, state_all, proj_all, hist_all):
    """Advance all layers by n_dev[0] rows: qk_all (L, Tmax, 32, 128), v_all (L, Tmax, 48, 128),
    gb_all (L, Tmax, 2, 48), state_all (L, 48, 128, 128), proj_all (L, Tmax, PROJ), hist_all (L, 3, 10240)."""
    Lyr, Tmax = qk_all.shape[0], qk_all.shape[1]
    _check(lib.gdn_commit(Lyr, Tmax, p(n_dev), p(qk_all), p(v_all), p(gb_all), p(state_all), p(proj_all),
                          proj_all.stride(1), proj_all.stride(0), p(hist_all), stream()), "gdn_commit")


lib.add_rmsnorm.argtypes = [I, I, P, P, P, P, I, I, P]
lib.attn_prep.argtypes = [I, P, P, P, P, P, P, P, P]
lib.silu_mul.argtypes = [I, I, P, P, P]
lib.gate_mul.argtypes = [I, P, P, P]


def add_rmsnorm(x, d, w, out):
    """x (R, H) bf16 (rows may be strided); x += d (if d is not None, contiguous); out = rmsnorm(x) * (1 + w)."""
    assert x.stride(1) == 1 and out.stride(1) == 1 and (d is None or d.is_contiguous())
    _check(lib.add_rmsnorm(x.shape[0], x.shape[1], p(x), P(0) if d is None else p(d), p(w), p(out), x.stride(0),
                           out.stride(0), stream()), "add_rmsnorm")


def attn_prep(qkv, qw, kw, pos_dev, qout, kc, vc):
    """qkv (R, 14336); writes qout (R, 24, 256) and the cache rows pos..pos+R-1 of kc, vc (L, 4, 256)."""
    _check(lib.attn_prep(qkv.shape[0], p(qkv), p(qw), p(kw), p(pos_dev), p(qout), p(kc), p(vc), stream()),
           "attn_prep")


def silu_mul(gu, act):
    _check(lib.silu_mul(gu.shape[0], act.shape[1], p(gu), p(act), stream()), "silu_mul")


def gate_mul(o, qkv):
    _check(lib.gate_mul(o.shape[0], p(o), p(qkv), stream()), "gate_mul")


lib.gated_norm.argtypes = [I, P, P, I, P, P, P]
lib.conv_update.argtypes = [P, P, I, P, P]


lib.gated_norm_bf16.argtypes = [I, P, P, I, P, P, P]


def gated_norm(o, proj, gw, out):
    """o (T, 48, 128) fp32 or bf16 -> out (T, 6144) bf16, gated by z = proj[:, 10240:16384]."""
    assert o.is_contiguous() and out.is_contiguous()
    f = lib.gated_norm_bf16 if o.dtype == torch.bfloat16 else lib.gated_norm
    _check(f(o.shape[0], p(o), p(proj), proj.stride(0), p(gw), p(out), stream()), "gated_norm")


def conv_update(n_dev, proj, hist):
    """hist (3, 10240) <- rows n, n+1, n+2 of [hist; proj rows] (n = n_dev[0] = number of proj rows)."""
    _check(lib.conv_update(p(n_dev), p(proj), proj.stride(0), p(hist), stream()), "conv_update")


lib.quant_fp8.argtypes = [I, I, P, I, P, P, P]


def quant_fp8(x, q, sc):
    """x (R, K) bf16 (K % 8 == 0, rows may be strided) -> q (R, K) float8_e4m3fn, sc (R, 1) fp32 row scales."""
    assert x.stride(1) == 1 and x.shape[1] % 8 == 0 and q.is_contiguous()
    _check(lib.quant_fp8(x.shape[0], x.shape[1], p(x), x.stride(0), p(q), p(sc), stream()), "quant_fp8")


lib.gdn_conv.argtypes = [I, P, I, P, P, P, P, P, P, P]


def gdn_conv(proj, hist, cw, aneg, dtb, y, gb):
    """Prefill: y (T, 10240) bf16 = silu(conv(q|k|v)), raw; gb (T, 2, 48) fp32 = g, beta."""
    T = proj.shape[0]
    _check(lib.gdn_conv(T, p(proj), proj.stride(0), p(hist), p(cw), p(aneg), p(dtb), p(y), p(gb), stream()),
           "gdn_conv")


lib.silu_mul_quant.argtypes = [I, I, P, P, P, P]


def silu_mul_quant(gu, q, sc):
    """gu (R, 2I) bf16 contiguous -> q (R, I) fp8 = silu(gate) * up quantized per row, sc (R, 1) fp32."""
    assert gu.is_contiguous() and q.is_contiguous()
    _check(lib.silu_mul_quant(gu.shape[0], gu.shape[1] // 2, p(gu), p(q), p(sc), stream()), "silu_mul_quant")


lib.add_rmsnorm_q.argtypes = [I, I, P, P, P, P, P, P]


def add_rmsnorm_q(x, d, w, q, sc):
    """add_rmsnorm with the output quantized: x (R, H) contiguous; q (R, H) fp8, sc (R, 1) fp32."""
    assert x.is_contiguous() and q.is_contiguous() and (d is None or d.is_contiguous()) and x.shape[1] == 5120
    _check(lib.add_rmsnorm_q(x.shape[0], x.shape[1], p(x), P(0) if d is None else p(d), p(w), p(q), p(sc),
                             stream()), "add_rmsnorm_q")


lib.argmax_rows.argtypes = [I, I, I, P, P, P, P, P, P]
_AM_WS = {}


def argmax_rows(x, out, map=None, nb=None):
    """out (R,) int64 = argmax of each row of x (R, V) bf16 contiguous (first index on ties), through map if given."""
    R, V = x.shape
    nb = nb or max(1, min(64, V // 4096))
    if x.device not in _AM_WS:
        _AM_WS[x.device] = (torch.empty(16 * 64, dtype=torch.int64, device=x.device),
                            torch.zeros(16, dtype=torch.int32, device=x.device))
    ws, cnt = _AM_WS[x.device]
    assert x.is_contiguous() and R <= 16 and out.dtype == torch.int64
    _check(lib.argmax_rows(R, V, nb, p(x), P(0) if map is None else p(map), p(out), p(ws), p(cnt), stream()),
           "argmax_rows")


lib.mtp_in.argtypes = [I, I, P, P, P, I, P, P, P, P]


def mtp_in(tok, embed, hid, we, wh, mx):
    """mx (R, 2H) = [rmsnorm(embed[tok]) * (1 + we) | rmsnorm(hid) * (1 + wh)]; tok (R,) int64, hid rows strided."""
    R, Hd = hid.shape
    assert tok.dtype == torch.int64 and hid.stride(1) == 1 and mx.is_contiguous()
    _check(lib.mtp_in(R, Hd, p(tok), p(embed), p(hid), hid.stride(0), p(we), p(wh), p(mx), stream()), "mtp_in")


lib.accept.argtypes = [I, P, P, P, P, P, P, P, P, P]


def accept(vt, y, n_acc, new, res, pick, pos, P_):
    """One round's acceptance and position bookkeeping (see accept_kernel in small.hip)."""
    _check(lib.accept(vt.shape[0], p(vt), p(y), p(n_acc), p(new), p(res), p(pick), p(pos), p(P_), stream()), "accept")
