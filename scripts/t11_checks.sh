#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T11a 自适应规则检查 A1–A3（结果 results/t11_checks/），最后运行 scripts/t11_checks.py analyze。
# 三条链并行（WSL 内存 9 GB，最多 3 个评估进程）：
#  A) CPU 渲染：A1 参照 k=100（种子 0–9，B=10，--log-traj）
#  B) CPU 渲染：A1 adaptive m=10, tau=inf（同上）
#  C) GPU：A2 tau=0, min_gap=m（m=10 / m=20，种子 0–9，--log-traj），然后 A3 m=10, tau=inf（种子 0–19，无扰动 / 推 0.06）
set -o pipefail
source scripts/env.sh
OUT=results/t11_checks
mkdir -p $OUT
FILTER="UserWarning|logger.warn|task_description|deprecated|warnings.warn"
G="python scripts/eval.py --config configs/act_default.yaml --batch-size 10"
AD="--mode adaptive --adapt-h-cmp 20"

( $G --seeds 0-9 $AD --adapt-m 10 --adapt-tau 0 --adapt-min-gap 10 --log-traj --out $OUT/a2_m10 \
  && $G --seeds 0-9 $AD --adapt-m 20 --adapt-tau 0 --adapt-min-gap 20 --log-traj --out $OUT/a2_m20 \
  && $G --seeds 0-19 $AD --adapt-m 10 --adapt-tau inf --adapt-min-gap 20 --out $OUT/a3_none \
  && python scripts/eval.py --config configs/act_push.yaml --batch-size 10 --seeds 0-19 --perturb-strength 0.06 \
       $AD --adapt-m 10 --adapt-tau inf --adapt-min-gap 20 --out $OUT/a3_push0.06 ) 2>&1 | grep -v -E "$FILTER" > $OUT/chain_c.log &
PC=$!

export CHUNK_RENDER=cpu
source scripts/env.sh
export LP_NUM_THREADS=7
C="python scripts/eval.py --config configs/act_default.yaml --seeds 0-9 --batch-size 10 --allow-cpu-render --log-traj"
( $C --k 100 --out $OUT/a1_k100 ) 2>&1 | grep -v -E "$FILTER" > $OUT/chain_a.log &
PA=$!
( $C $AD --adapt-m 10 --adapt-tau inf --adapt-min-gap 20 --out $OUT/a1_adaptive_inf ) 2>&1 | grep -v -E "$FILTER" > $OUT/chain_b.log &
PB=$!

fail=0
for p in $PA $PB $PC; do wait $p || fail=1; done
grep -ah "批 \|Error\|error" $OUT/chain_*.log | tail -40
for d in $OUT/a1_* $OUT/a2_* $OUT/a3_*; do python scripts/summarize.py $d > /dev/null || { echo "summarize 校验失败：$d"; fail=1; }; done
python scripts/t11_checks.py analyze || fail=1
exit $fail
