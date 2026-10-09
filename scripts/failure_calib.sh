#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T13：失败分类阈值的标定数据（调试种子 0–9，GPU，B=10，--log-traj，含 T13 新增的指尖位置）。约 5 分钟。
# 输出 results/failure_calib/<格子名>/；阈值由 scripts/analyze_failure.py calib 读这些数据算出，写进 results/failure/definitions.md。
#   wsl.exe -d Ubuntu-24.04 -u root --cd /mnt/d/xiangmu/chunk-reobserve -- bash scripts/failure_calib.sh
set -eo pipefail
source scripts/env.sh
OUT=results/failure_calib
mkdir -p "$OUT"
filter() { { grep --line-buffered -v -E "UserWarning|logger.warn|task_description|deprecated|^\s*[\{\}\"]|^  " || true; }; }
run() {   # 名字 k 扰动参数...
  local name=$1 k=$2; shift 2
  python scripts/eval.py --config configs/act_default.yaml --out "$OUT/$name" --k "$k" --seeds 0-9 --batch-size 10 \
    --log-traj "$@" 2>&1 | filter
}
run k100__none 100 &
run k20__none 20 &
wait
run k100__push0.06 100 --perturb push --perturb-strength 0.06 --perturb-param t_lo=20 --perturb-param t_hi=100 &
run k20__push0.06 20 --perturb push --perturb-strength 0.06 --perturb-param t_lo=20 --perturb-param t_hi=100 &
wait
run k20__delay10 20 --perturb action_delay --perturb-strength 10 &
run k100__delay10 100 --perturb action_delay --perturb-strength 10 &
wait
echo "=== 标定数据完成：$OUT"
