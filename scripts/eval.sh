#!/bin/bash
# 统一评估包装：bash scripts/eval.sh --config configs/act_default.yaml --out results/<实验名> [其他参数]
# 从 Windows：wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/d/xiangmu/chunk-reobserve/scripts/eval.sh --config ... --out ...
set -eo pipefail
source /mnt/d/xiangmu/chunk-reobserve/scripts/env.sh
python scripts/eval.py "$@"
