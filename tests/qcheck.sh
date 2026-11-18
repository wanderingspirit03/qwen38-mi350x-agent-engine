#!/bin/bash
# tests/qcheck.sh [PROMPTS]: ./check (answers, the reference's walk: measure A; public tasks: measure B) on this
# repo's agentic prompts instead of the 8 short public ones. Run under ./try.
cd "$(dirname "$0")/.."
P="$(realpath "${1:-tests/qprompts.jsonl}")"
CFG=/tmp/qcheck-$$.json
python3 - "$P" "$CFG" <<'EOF'
import json, sys
A = "/opt/ij-amd/amd/"
c = json.load(open(A + "config.json"))
c["check"]["prompts"] = sys.argv[1]
c["quality"]["tasks"]["check_file"] = A + c["quality"]["tasks"]["check_file"]
for k in ("start", "stop"):
    c["ref"][k][0] = A + c["ref"][k][0]
json.dump(c, open(sys.argv[2], "w"))
EOF
IJ_AMD_CONFIG=$CFG /opt/ij-amd/amd/check "$PWD"
rc=$?
rm -f $CFG
exit $rc
