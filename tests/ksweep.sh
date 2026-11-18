#!/bin/bash
# tests/ksweep.sh N K...: tests/run.sh with each draft length K (acceptance and speed per K)
cd "$(dirname "$0")/.."
N=$1; shift
for k in "$@"; do
  echo "=== K=$k"
  ENG_SPEC_K=$k tests/run.sh $N 128 | grep -v '^ANS' | grep -E 'TOTAL|gen:|ready'
done
