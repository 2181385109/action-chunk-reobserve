#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T9 分析脚本演示运行（不是结果）：DP 用试跑数据 results/pilot_dp/（种子 5000–5049），ACT 的差距变化用已有的 results/main/。
# 输出 results/pilot_dp/analyze_dryrun/{main_dp,interaction}/。
source scripts/env.sh
python scripts/analyze_main_dp.py --dir results/pilot_dp --out results/pilot_dp/analyze_dryrun/main_dp --seeds 5000-5049
python scripts/analyze_interaction.py --dp-dir results/pilot_dp --dp-seeds 5000-5049 --out results/pilot_dp/analyze_dryrun/interaction
