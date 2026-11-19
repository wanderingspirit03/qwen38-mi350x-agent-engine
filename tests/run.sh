#!/bin/bash
# tests/run.sh [N] [MAX_TOKENS] [PROMPTS]: build the image, start ./serve, send prompts, stop. Run under ./try.
set -uo pipefail
cd "$(dirname "$0")/.."
N="${1:-4}"; MT="${2:-128}"; P="${3:-tests/prompts.jsonl}"
PORT=8123
sudo -n /opt/ij-amd/amd/controller/box/ij-docker build . > /tmp/eng-build.log 2>&1 || { tail -30 /tmp/eng-build.log; exit 1; }
./serve $PORT > /tmp/eng-serve.log 2>&1 &
SP=$!
t0=$(date +%s)
until curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null; do
  sleep 2
  if ! kill -0 $SP 2>/dev/null; then echo "serve died"; tail -50 /tmp/eng-serve.log; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt 1800 ]; then echo "not ready"; break; fi
done
echo "ready after $(( $(date +%s) - t0 )) s"
python3 tests/client.py $PORT "$P" "$N" "$MT"
kill $SP; wait $SP 2>/dev/null
echo "--- serve log"; grep -E '^(gen|rounds):' /tmp/eng-serve.log; tail -5 /tmp/eng-serve.log
