"""Compile-only (no GPU): python3 -m eng.kernels.attn2_asm VARIANT_JSON...  prints registers/spills/LDS per variant."""
import json
import re
import sys
from collections import Counter

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

TARGET = GPUTarget("hip", "gfx950", 64)


def compile_kernel(fn, types, constexprs, num_warps=4, num_stages=2, **opts):
    sig = {}
    attrs = {}
    for i, name in enumerate(fn.arg_names):
        if name in constexprs:
            sig[name] = "constexpr"
        else:
            t = types.get(name, "i32")
            sig[name] = t
            if t.startswith("*") or name.startswith("s"):
                attrs[(i,)] = [["tt.divisibility", 16]]
    src = ASTSource(fn, sig, constexprs=constexprs, attrs=attrs)
    return triton.compile(src, target=TARGET, options=dict(num_warps=num_warps, num_stages=num_stages, **opts))


def summary(k, show_loop=False):
    asm = k.asm["amdgcn"]
    out = {}
    for key in ["NumVgprs", "NumAgprs", "TotalNumVgprs", "ScratchSize", "Occupancy", "NumSgprs"]:
        m = re.search(r";\s*" + key + r":\s*(\d+)", asm)
        if m:
            out[key] = int(m.group(1))
    out["lds"] = k.metadata.shared
    ins = Counter()
    for line in asm.splitlines():
        s = line.strip()
        if not s or s.startswith((";", ".", "/")) or s.endswith(":"):
            continue
        op = s.split()[0]
        if "mfma" in op:
            ins["mfma"] += 1
        elif op.startswith("v_accvgpr"):
            ins["accmov"] += 1
        elif op.startswith(("ds_read", "ds_load")):
            ins["ds_read"] += 1
        elif op.startswith(("ds_write", "ds_store")):
            ins["ds_write"] += 1
        elif op.startswith(("global_load", "buffer_load")):
            ins["gload"] += 1
        elif op.startswith("v_exp"):
            ins["exp"] += 1
        elif op.startswith("v_"):
            ins["valu"] += 1
        elif op.startswith("s_waitcnt"):
            ins["wait"] += 1
    out["ins"] = dict(ins)
    return out


def blocks(k):
    asm = k.asm["amdgcn"]
    res = []
    cur, cnt = "entry", Counter()
    for line in asm.splitlines():
        s = line.strip()
        if re.match(r"^\.?LBB\d+_\d+:", s) or re.match(r"^\.L\w+:", s):
            if cnt["mfma"]:
                res.append((cur, dict(cnt)))
            cur, cnt = s[:-1], Counter()
            continue
        if not s or s.startswith((";", ".", "/")) or s.endswith(":"):
            continue
        op = s.split()[0]
        if "mfma" in op:
            cnt["mfma"] += 1
        elif op.startswith("v_accvgpr"):
            cnt["accmov"] += 1
        elif op.startswith(("ds_read", "ds_load")):
            cnt["ds_read"] += 1
        elif op.startswith(("ds_write", "ds_store")):
            cnt["ds_write"] += 1
        elif op.startswith(("global_load", "buffer_load")):
            cnt["gload"] += 1
        elif op.startswith("v_exp"):
            cnt["exp"] += 1
        elif op.startswith("v_pk"):
            cnt["vpk"] += 1
        elif op.startswith("v_"):
            cnt["valu"] += 1
            cnt["v:" + op] += 1
        elif op.startswith("s_waitcnt"):
            cnt["wait"] += 1
        elif op.startswith("s_"):
            cnt["salu"] += 1
    if cnt["mfma"]:
        res.append((cur, dict(cnt)))
    return res
