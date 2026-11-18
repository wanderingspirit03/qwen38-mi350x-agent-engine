"""Compile a .hip file for gfx950 and print a compressed view of its device assembly (tool, not engine code).
Run: python3 -m eng.kernels.asm_summary eng/csrc/attn_prefill.hip [kernel-substring]"""
import collections, re, subprocess, sys

def cls(ins):
    op = ins.split()[0]
    if op.startswith("v_mfma"): return "mfma"
    if op.startswith("ds_read") or op.startswith("ds_load"): return op
    if op.startswith("ds_write") or op.startswith("ds_store"): return op
    if op.startswith("global_load_lds") or op.startswith("buffer_load") or op.startswith("global_load"): return op
    if op.startswith("v_exp"): return "v_exp"
    if op in ("s_waitcnt", "s_barrier") or op.startswith("s_cbranch") or op == "s_branch": return ins
    if op.startswith("scratch") or op.startswith("buffer_store"): return op
    if op.startswith("v_accvgpr"): return "accmov"
    if op.startswith("v_"): return "valu"
    return None

def main():
    src = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else ""
    flags = [a for a in sys.argv[3:] if a.startswith("-D")]
    subprocess.run(["/opt/rocm/bin/hipcc", "-O3", "--offload-arch=gfx950", "--cuda-device-only", "-S", "-o", "/tmp/a.s", *flags, src],
                   check=True, stderr=subprocess.DEVNULL)
    lines = open("/tmp/a.s").read().splitlines()
    inside = False
    out, last, cnt = [], None, 0
    def flush():
        if last is not None:
            out.append(f"{last} x{cnt}" if cnt > 1 else last)
    for l in lines:
        st = l.strip()
        if re.match(r"^[A-Za-z_]\S*:", l):
            inside = want in st
            if inside:
                flush(); last, cnt = None, 0
                out.append("== " + st)
            continue
        if not inside:
            continue
        if re.match(r"^\.LBB\S*:", st) or st.startswith("s_endpgm"):
            flush(); last, cnt = None, 0
            out.append(st)
            continue
        if not st or st.startswith((";", ".")):
            continue
        c = cls(st.split(";")[0].strip())
        if c is None:
            continue
        if c == last:
            cnt += 1
        else:
            flush(); last, cnt = c, 1
        if st.startswith("s_endpgm"):
            break
    flush()
    if "full" in sys.argv:
        print("\n".join(out))
    if "blocks" in sys.argv:   # per basic block: counts of instruction classes
        blk, cur = [], None
        for o in out:
            if o.startswith(".LBB") or o.startswith("=="):
                cur = [o.split()[0], collections.Counter()]
                blk.append(cur)
                continue
            if cur is None:
                continue
            k = o.split(" x")[0]
            k = "wait" if k.startswith("s_waitcnt") else ("branch" if k.startswith("s_cbranch") or k.startswith("s_branch") else k)
            cur[1][k] += int(o.split(" x")[1]) if " x" in o and o.split(" x")[1].isdigit() else 1
        for name, c in blk:
            print(name, dict(c.most_common()))
    print(collections.Counter(o.split(" x")[0] for o in out for _ in range(int(o.split(" x")[1]) if " x" in o else 1)).most_common(25))
    for l in lines:
        if any(k in l for k in (".vgpr_count", ".agpr_count", "spill_count", "lds_size", "ScratchSize")):
            print(l.strip())

if __name__ == "__main__":
    main()
