#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T4 扰动框架检查（约 15 分钟；结果 results/t4_checks/，说明见 scripts/t4_checks.py）。三条链并行（WSL 内存 9 GB，最多 3 个评估进程）：
#  A) GPU：P3 k=100 d=0 / d=D_MAX（后者 --log-traj，种子 0、1 每步存画面 = P2）/ te d=0
#  B) GPU：P3 k=10 d=0 / d=D_MAX / te d=D_MAX
#  C) CPU 渲染：P1 不开扰动 vs push d=0（种子 0–9，k=100，--log-traj）
set -o pipefail
source scripts/env.sh
OUT=results/t4_checks
DMAX=0.10
mkdir -p $OUT
FILTER="UserWarning|logger.warn|task_description|deprecated|warnings.warn"
EV="python scripts/eval.py --seeds 0-19 --batch-size 10"
PUSH="--config configs/act_push.yaml"

( $EV $PUSH --k 100 --perturb-strength 0 --out $OUT/p3_k100_d0 \
  && $EV $PUSH --k 100 --perturb-strength $DMAX --log-traj --save-frames 0,1 --frame-every 1 --out $OUT/p3_k100_dmax \
  && $EV $PUSH --mode te --perturb-strength 0 --out $OUT/p3_kte_d0 ) 2>&1 | grep -v -E "$FILTER" > $OUT/chain_a.log &
PA=$!
( $EV $PUSH --k 10 --perturb-strength 0 --out $OUT/p3_k10_d0 \
  && $EV $PUSH --k 10 --perturb-strength $DMAX --out $OUT/p3_k10_dmax \
  && $EV $PUSH --mode te --perturb-strength $DMAX --out $OUT/p3_kte_dmax ) 2>&1 | grep -v -E "$FILTER" > $OUT/chain_b.log &
PB=$!

export CHUNK_RENDER=cpu
source scripts/env.sh
export LP_NUM_THREADS=10
( python scripts/eval.py --config configs/act_default.yaml --k 100 --seeds 0-9 --batch-size 10 --allow-cpu-render \
    --log-traj --out $OUT/p1_none \
  && python scripts/eval.py --config configs/act_push.yaml --k 100 --perturb-strength 0 --seeds 0-9 --batch-size 10 \
    --allow-cpu-render --log-traj --out $OUT/p1_d0 ) 2>&1 | grep -v -E "$FILTER" > $OUT/chain_c.log &
PC=$!

fail=0
for p in $PA $PB $PC; do wait $p || fail=1; done
grep -ah "批 \|Error\|error" $OUT/chain_*.log | tail -30
for d in $OUT/p1_* $OUT/p3_*; do python scripts/summarize.py $d > /dev/null || { echo "summarize 校验失败：$d"; fail=1; }; done
python scripts/t4_checks.py analyze || fail=1
exit $fail
