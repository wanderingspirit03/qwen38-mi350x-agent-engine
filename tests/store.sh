#!/bin/bash
# tests/store.sh CMD...: run CMD (e.g. tests/in_container.sh python3 -m eng.mtp_train ...) with the file store of
# tests/store_server.py at 127.0.0.1:8765 over /home/builder/work/mtpstore (kept between jobs; eng/store.py)
cd "$(dirname "$0")/.."
python3 tests/store_server.py /home/builder/work/mtpstore 8765 &
S=$!
trap "kill $S" EXIT
sleep 1
"$@"
