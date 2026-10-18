#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python -u "$root/run_all.py" --data-dir "${1:?DATA_ROOT containing training2017}" --device "${2:-cuda:0}" --epochs 100 --batch-size 16 --window-length 9000 --repeat-times 3 --seeds 42 43 44 45 46 --suite main --out-root "$root/runs/cinc2017_paired_seeds" "${@:3}"
