#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T8c K1/K2: DP k=4 vs lerobot's own eval, and k=4 vs k=8 trajectory statistics. Results: results/t8_dp_k4/.
# Runs sequentially (no GPU contention): ours k=4 (--log-traj) ; lerobot k=4 ; lerobot k=8. ~10 min total.
# k=8 "ours" = results/t8_dp_baseline/run_k8 (no perturbation, --log-traj).
set -o pipefail
source scripts/env.sh
OUT=results/t8_dp_k4
mkdir -p $OUT
FILTER="UserWarning|logger.warn|task_description|deprecated|warnings.warn|Running rollout|Stepping through"
python scripts/eval.py --config configs/dp_default.yaml --seeds 0-49 --batch-size 50 --k 4 --log-traj --out $OUT/ours_k4 2>&1 | grep -v -E "$FILTER" > $OUT/ours_k4.log
python scripts/summarize.py $OUT/ours_k4 > /dev/null || exit 1
python scripts/t8_dp_k4.py lerobot --k 4 2>&1 | grep -v -E "$FILTER" > $OUT/lerobot_k4.log
python scripts/t8_dp_k4.py lerobot --k 8 2>&1 | grep -v -E "$FILTER" > $OUT/lerobot_k8.log
python scripts/t8_dp_k4.py analyze 2>&1 | grep -v -E "$FILTER"
