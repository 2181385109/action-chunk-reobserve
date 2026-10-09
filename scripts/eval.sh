#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# 统一评估包装：bash scripts/eval.sh --config configs/act_default.yaml --out results/<实验名> [其他参数]
# 从 Windows：wsl.exe -d Ubuntu-24.04 -u root -- bash <repo>/scripts/eval.sh --config ... --out ...
set -eo pipefail
source scripts/env.sh
python scripts/eval.py "$@"
