#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${1:?Usage: bash run_cpsc2018.sh WFDB_ROOT [cuda:0] [options]}"
shift
device="cuda:0"
if [[ $# -gt 0 && "$1" != --* ]]; then device="$1"; shift; fi
exec python -u "$project_dir/run_cpsc2018.py" --data-dir "$data_root" --device "$device" "$@"
