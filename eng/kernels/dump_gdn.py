"""Print register stats of the gdn_prefill kernels; --asm NAME prints that kernel's assembly.
python3 -m eng.kernels.dump_gdn [--asm _scan_kernel] [kw=val ...]"""
import re, sys
import torch
from eng.kernels import gdn_prefill as M
from eng.kernels.test_gdn_prefill import make_inputs

kw = {}
want = None
args = sys.argv[1:]
if "--asm" in args:
    want = args[args.index("--asm") + 1]
    args = [a for a in args if a not in ("--asm", want)]
for a in args:
    key, val = a.split("=")
    kw[key] = int(val)
T = 1000
q, k, v, g, beta, S0 = make_inputs(T)
M.gdn_prefill(q, k, v, g, beta, S0, **kw)
torch.cuda.synchronize()
for kern in (M._wy_kernel, M._scan_kernel, M._out_kernel):
    for dev_cache in kern.device_caches.values():
        for ck in dev_cache[0].values():
            asm = ck.asm["amdgcn"]
            pat = ["v_mfma", "scratch_", "buffer_load", "global_load", "buffer_store", "global_store", "ds_read", "ds_write", "s_barrier", "s_waitcnt", "v_exp"]
            cnt = {p: len(re.findall(r"^\s*" + p, asm, re.M)) for p in pat}
            meta = re.findall(r"^\s*\.(sgpr_count|vgpr_count|agpr_count|group_segment_fixed_size|private_segment_fixed_size|vgpr_spill_count|sgpr_spill_count):\s*(\d+)", asm, re.M)
            print(kern.fn.__name__, dict(meta), cnt, "lines", asm.count("\n"))
            if kern.fn.__name__ == want:
                print("=====ASM=====")
                print(asm)
                print("=====END=====")
