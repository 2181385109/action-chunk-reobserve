#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T11a：run_tune_adaptive.sh + select_adaptive.py 的极小配置自检（种子 0–1，B=2，每回合最多 30 步，输出在 WSL /tmp，不进仓库）。
# 第一次跑 1-3 格后中断式地停下，再跑全部（验证续跑跳过已完成的格子），最后自动运行选择规则。
set -eo pipefail
export TUNE_OUT=/tmp/t11_tune_smoke/results TUNE_CONFIGS=/tmp/t11_tune_smoke/configs TUNE_SEEDS=0-1 TUNE_BATCH=2 TUNE_MAX_STEPS=30
rm -rf /tmp/t11_tune_smoke
bash scripts/run_tune_adaptive.sh 1-3
bash scripts/run_tune_adaptive.sh
test -f "$TUNE_OUT/selection.json"
echo "=== 自检通过：$(python3 -c "import json; print(json.load(open('$TUNE_OUT/selection.json'))['verdict'])")"
