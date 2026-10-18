#!/usr/bin/env bash
set -euo pipefail
# 相对数据路径按调用者所在目录解析，不切换 cwd。
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${1:?用法: bash run_all.sh DATA_ROOT [cuda:0] [其他参数]}"
shift
device="cuda:0"
if [[ $# -gt 0 && "$1" != --* ]]; then device="$1"; shift; fi
exec python -u "$project_dir/run_all.py" --data-dir "$data_root" --device "$device" "$@"
