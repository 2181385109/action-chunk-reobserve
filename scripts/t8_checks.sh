#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T8b checks D1-D3 (DP / PushT push-T; results in results/t8_checks/, see scripts/t8_checks.py). ~15 min, 3 parallel chains.
#  A) D1: k=8 push d=0 (--log-traj) ; then D3 k=15 none / d=DMAX
#  B) D2: k=8 d=DMAX (--log-traj, frames for seeds 0-3 every step) ; then D3 k=4 d=DMAX
#  C) D3: k=4 none
# Baseline (no perturbation, k=8, seeds 0-49, --log-traj) = results/t8_dp_baseline/run_k8 (step 1).
set -o pipefail
source scripts/env.sh
OUT=results/t8_checks
DMAX=60
mkdir -p $OUT
FILTER="UserWarning|logger.warn|task_description|deprecated|warnings.warn"
EV="python scripts/eval.py --config configs/dp_push.yaml --seeds 0-49 --batch-size 50"
NONE="python scripts/eval.py --config configs/dp_default.yaml --seeds 0-49 --batch-size 50"

( $EV --k 8 --perturb-strength 0 --log-traj --out $OUT/d1_d0 \
  && $NONE --k 15 --out $OUT/d3_k15_none \
  && $EV --k 15 --perturb-strength $DMAX --out $OUT/d3_k15_dmax ) 2>&1 | grep -v -E "$FILTER" > $OUT/chain_a.log &
PA=$!
( $EV --k 8 --perturb-strength $DMAX --log-traj --save-frames 0,1,2,3 --frame-every 1 --out $OUT/d2_k8_dmax \
  && $EV --k 4 --perturb-strength $DMAX --out $OUT/d3_k4_dmax ) 2>&1 | grep -v -E "$FILTER" > $OUT/chain_b.log &
PB=$!
( $NONE --k 4 --out $OUT/d3_k4_none ) 2>&1 | grep -v -E "$FILTER" > $OUT/chain_c.log &
PC=$!

fail=0
for p in $PA $PB $PC; do wait $p || fail=1; done
grep -ah "批 \|Error\|error" $OUT/chain_*.log | tail -30
for d in $OUT/d1_* $OUT/d2_* $OUT/d3_*; do python scripts/summarize.py $d > /dev/null || { echo "summarize check failed: $d"; fail=1; }; done
python scripts/t8_checks.py analyze || fail=1
exit $fail
