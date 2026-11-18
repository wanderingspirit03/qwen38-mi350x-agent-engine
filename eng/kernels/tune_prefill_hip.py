"""Sweep compile-time variants of the experimental HIP prefill kernel eng/csrc/experimental/attn_prefill.hip.
Run: python3 -m eng.kernels.tune_prefill_hip "-DKPF=6 -DVPF=3" "-DKPF=8 -DVPF=4" ..."""
import ctypes, os, subprocess, sys

import torch

from .attention import attn_prefill
from .test_attention import make_cache, make_q, bench, prefill_flops, ref_attn, err

SRC = os.path.join(os.path.dirname(__file__), "..", "csrc", "experimental", "attn_prefill.hip")
P, I = ctypes.c_void_p, ctypes.c_int


def build(flags, i):
    so = f"/tmp/apf_{i}.so"
    r = subprocess.run(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx950", "-shared", "-fPIC", *flags.split(),
                        "-Rpass-analysis=kernel-resource-usage", "-o", so, SRC], check=True, capture_output=True, text=True)
    use = [l.split("remark:")[-1].strip() for l in r.stderr.splitlines()
           if "remark" in l and any(k in l for k in ("VGPRs:", "AGPRs:", "Spill", "Occupancy", "LDS Size"))]
    print(f"[{flags}] " + "; ".join(use), flush=True)
    lib = ctypes.CDLL(so)
    lib.attn_prefill.argtypes = [P, P, P, P, P, I, I, I, I, I, I, I, I, P]
    return lib


def run(lib, q, k, v, p0, gate, o):
    T = q.shape[0]
    pp = lambda t: P(t.data_ptr())
    rc = lib.attn_prefill(pp(q), pp(k), pp(v), pp(o), pp(gate), p0, T, q.stride(0), q.stride(1), o.stride(0),
                          o.stride(1), gate.stride(0), gate.stride(1), P(torch.cuda.current_stream().cuda_stream))
    assert rc == 0
    return o


def main():
    variants = sys.argv[1:] or [""]
    libs = [build(f, i) for i, f in enumerate(variants)]
    for T, p0 in [(4096, 6400), (1500, 6400), (5600, 6400)]:
        k, v = make_cache(p0 + T)
        q, gate = make_q(T)
        o = torch.empty(T, 24, 256, dtype=torch.bfloat16, device=q.device)
        tf = prefill_flops(T, p0)
        ref = ref_attn(q, k, v, p0 + T) * torch.sigmoid(gate.float()) if T == 1500 else None
        tt = bench(lambda: attn_prefill(q, k, v, p0, gate=gate), iters=10, warm=2)
        print(f"T={T} p0={p0} Triton: {tt:8.1f} us {tf / tt / 1e6:6.1f} TFLOP/s", flush=True)
        for f, lib in zip(variants, libs):
            us = bench(lambda: run(lib, q, k, v, p0, gate, o), iters=10, warm=2)
            e = f" err {err(run(lib, q, k, v, p0, gate, o), ref)[0]:.1e}" if ref is not None else ""
            print(f"T={T} [{f}]: {us:8.1f} us {tf / us / 1e6:6.1f} TFLOP/s{e}", flush=True)


if __name__ == "__main__":
    main()
