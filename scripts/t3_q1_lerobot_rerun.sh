#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# 用法：bash scripts/t3_q1_lerobot_rerun.sh；EPLEN=100 时只跑前 100 步并保存动作（逐步逐位比较用）
# 只重跑 T3 Q1 的 lerobot 一侧（CPU 渲染）。第一次运行时 eval_main 写 eval_info.json 失败（不录视频时它不建输出目录），已修复。
export CHUNK_RENDER=cpu
source scripts/env.sh
export LP_NUM_THREADS=6
python scripts/t3_diag.py q1-lerobot ${EPLEN:+--episode-length $EPLEN} 2>&1 | grep -v -E "UserWarning|logger.warn|task_description|deprecated|warnings.warn|Running rollout|Stepping through" > results/t3_k1_diag/q1_lerobot${EPLEN:+_len$EPLEN}.log
