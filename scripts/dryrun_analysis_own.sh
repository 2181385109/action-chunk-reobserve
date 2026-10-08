#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T10b 分析脚本演示运行（DRY RUN，不是结果）：把预训练模型的对应 9 格 results/main/ 和基础水平复现 results/repro/act 当作
# “自训练模型”的数据喂给 analyze_main_own.py（不评估自训练模型）。输出 results/main_own_dryrun/。
source /mnt/d/xiangmu/chunk-reobserve/scripts/env.sh
python scripts/analyze_main_own.py --dir results/main --baseline-dir results/repro/act --out results/main_own_dryrun --dry-run
