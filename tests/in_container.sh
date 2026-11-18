#!/bin/bash
# tests/in_container.sh CMD...: build the image from this tree and run CMD inside it (GPU, /models read-only).
# Run it under ./try:  ./try tests/in_container.sh python3 -m eng.some_test
cd "$(dirname "$0")/.."
L=/opt/ij-amd/amd/controller/box/ij-docker
sudo -n $L build . > /tmp/eng-build-$$.log 2>&1 || { tail -30 /tmp/eng-build-$$.log; exit 1; }
rm -f /tmp/eng-build-$$.log
exec sudo -n $L run --rm --shm-size 16g ij-builder/engine:latest "$@"
