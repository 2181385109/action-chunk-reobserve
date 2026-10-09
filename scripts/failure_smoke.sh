#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T13 冒烟：用 run_failure.sh 本身跑极小配置（种子 0–1、2 格、B=2，约 2 分钟），证明“跑数据 → 续跑跳过 → 自动报告”整条流程能跑。
# 输出 results/failure_smoke/（画面不入库）。不是结果。
set -eo pipefail
export FAIL_OUT=results/failure_smoke FAIL_SEEDS=0-1 FAIL_CELLS=k20__push0.06,k100__push0.06 FAIL_BATCH=2
export FAIL_LABEL="冒烟运行：种子 0–1、2 格、每格 2 回合，只证明流程能跑，不是结果"
rm -rf results/failure_smoke results/failure_smoke.log
bash scripts/run_failure.sh
echo "=== 再跑一次（应全部跳过并重新生成报告）"
bash scripts/run_failure.sh
