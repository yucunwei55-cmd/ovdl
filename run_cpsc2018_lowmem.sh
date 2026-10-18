#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python -u "$root/run_cpsc2018.py" --data-dir "${1:?WFDB_ROOT}" --device "${2:-cuda:0}" --epochs 100 --batch-size 32 --window-length 9000 --repeat-times 4 --preprocessing legacy --tail-mode pad --weight-decay 0 --seeds 42 43 44 45 46 --durations fixed variable --variants main --out-root "$root/runs/cpsc2018_postnorm_aug_v4" "${@:3}"
