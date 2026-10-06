#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T3 ACT k=1 诊断（约 15–20 分钟；结果 results/t3_k1_diag/，说明见 scripts/t3_diag.py）。三条链并行：
#  A) GPU 渲染：Q2 剂量反应（k=1,2,5,10,100,te；种子 0–19；--log-traj；k=1/k=100 导出种子 0 画面 = Q4）
#  B) CPU 渲染：我们的脚本 k=1 种子 0–9，--log-traj 关 / 开（开时同时 --save-frames 0）各一次（Q1 + 记录不改结果）
#  C) CPU 渲染：lerobot 0.3.3 自带 eval_main，n_action_steps=1，种子 0–9（Q1）
set -o pipefail
source scripts/env.sh
OUT=results/t3_k1_diag
mkdir -p $OUT
FILTER="UserWarning|logger.warn|task_description|deprecated|warnings.warn"

python scripts/t3_diag.py q2 2>&1 | grep -v -E "$FILTER" > $OUT/q2.log &
PA=$!

export CHUNK_RENDER=cpu
source scripts/env.sh
export LP_NUM_THREADS=6   # 两个 CPU 渲染进程各 6 线程（16 核，留给 GPU 链和其他进程）
( python scripts/eval.py --config configs/act_default.yaml --k 1 --seeds 0-9 --batch-size 10 --allow-cpu-render \
    --out $OUT/q1_ours_cpu_k1 \
  && python scripts/eval.py --config configs/act_default.yaml --k 1 --seeds 0-9 --batch-size 10 --allow-cpu-render \
    --log-traj --save-frames 0 --out $OUT/q1_ours_cpu_k1_traj ) 2>&1 | grep -v -E "$FILTER" > $OUT/q1_ours.log &
PB=$!
python scripts/t3_diag.py q1-lerobot 2>&1 | grep -v -E "$FILTER" > $OUT/q1_lerobot.log &
PC=$!

fail=0
for p in $PA $PB $PC; do wait $p || fail=1; done
grep -ah "批 \|Q1\|Error\|error" $OUT/*.log | grep -v "^\s*$" | tail -30
python scripts/t3_diag.py analyze
cat $OUT/diag.md
exit $fail
