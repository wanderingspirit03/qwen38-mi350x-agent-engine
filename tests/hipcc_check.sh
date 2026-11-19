#!/bin/bash
# tests/hipcc_check.sh <module> : compile the SRC string of a python module for gfx950 on the box (errors + registers)
cd "$(dirname "$0")/.."
D=$(mktemp -d)
python3 -c "import importlib.util, sys; s = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)" 2>/dev/null
python3 - "$1" "$D" <<'EOF'
import re, sys
t = open(sys.argv[1]).read()
src = re.search(r'SRC = r"""(.*?)"""', t, re.S).group(1)
open(sys.argv[2] + "/k.hip", "w").write(src)
EOF
cd $D && /opt/rocm/bin/hipcc -O3 --offload-arch=gfx950 -shared -fPIC $HIPFLAGS -save-temps=obj -o k.so k.hip 2>&1 | grep -B2 -A6 error | head -40
grep -c "vgpr_spill_count:     0" $D/*gfx950*.s; grep "vgpr_spill_count:" $D/*gfx950*.s | grep -v ":     0" | head
[ -n "$DUMP" ] && cat $D/*gfx950*.s
ls -la $D/k.so; rm -rf $D
