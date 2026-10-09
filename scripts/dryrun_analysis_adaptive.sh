#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T11b 分析脚本演示运行（DRY RUN，不是结果）：用阈值试跑数据 results/tune_adaptive（调参种子 6000–6049）里的 ad_m10_q90__{none,push0.06}
# 和同种子的 k100__{none,push0.06} 当作“正式数据”喂给 analyze_main_adaptive.py（HA1/HA2 因此是配对的）；
# k20/k50/te 只有 results/main 里的（种子 200000–200199），和试跑种子不配对，报告里只列成功率 / 调用次数。输出 results/main_adaptive_dryrun/。
source scripts/env.sh
python scripts/analyze_main_adaptive.py --dir results/tune_adaptive --ad-policy ad_m10_q90 --control-dir results/tune_adaptive \
    --extra-control-dir results/main --seeds 6000-6049 --out results/main_adaptive_dryrun --dry-run
