#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${1:?Usage: bash run_cinc2017.sh DATA_ROOT [cuda:0] [options]}"
shift
device="cuda:0"
if [[ $# -gt 0 && "$1" != --* ]]; then device="$1"; shift; fi
# Delegate to the CinC2017 launcher with a dataset-specific output root.
exec bash "$project_dir/run_all.sh" "$data_root" "$device" --out-root "$project_dir/runs/cinc2017" "$@"
