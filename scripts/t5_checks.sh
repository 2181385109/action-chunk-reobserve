#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T5 噪声 / 延迟检查（N2 单独跑：bash scripts/t5_py.sh n2；本脚本跑 N1 / N3 / N4 并 analyze；结果 results/t5_checks/）。
# 三条链并行（WSL 内存 9 GB，最多 3 个评估进程）：
#  A) CPU 渲染：N1 aloha 不开扰动 / σ=0 / d=0（种子 0–9，k=100，--log-traj）
#  B) GPU：N4 噪声 σ=最大档 k=100 / k=10 / te（种子 0–19），然后 N1 DP 不开扰动 / σ=0
#  C) GPU：N4 延迟 d=最大档 k=100 / k=10 / te（--log-traj，给 N3），然后 N1 DP d=0、N3 DP d=最大档
set -o pipefail
source scripts/env.sh
OUT=results/t5_checks
SIG=16        # = t5_checks.SIGMA_LEVELS["act"][-1]
DA=10         # = t5_checks.D_LEVELS["act"][-1]
DD=4          # = t5_checks.D_LEVELS["dp"][-1]
mkdir -p $OUT
FILTER="UserWarning|logger.warn|task_description|deprecated|warnings.warn"
ACT="python scripts/eval.py --config configs/act_default.yaml --seeds 0-19 --batch-size 10"
DP="python scripts/eval.py --config configs/dp_default.yaml --seeds 0-9 --batch-size 10 --log-traj"
NZ="--perturb obs_noise --perturb-strength"
DL="--perturb action_delay --perturb-strength"

( $ACT --k 100 $NZ $SIG --out $OUT/n4_noise_100 \
  && $ACT --k 10 $NZ $SIG --out $OUT/n4_noise_10 \
  && $ACT --mode te $NZ $SIG --out $OUT/n4_noise_te \
  && $DP --out $OUT/n1_dp_none \
  && $DP $NZ 0 --out $OUT/n1_dp_noise0 ) 2>&1 | grep -v -E "$FILTER" > $OUT/chain_b.log &
PB=$!
( $ACT --k 100 $DL $DA --log-traj --out $OUT/n4_delay_100 \
  && $ACT --k 10 $DL $DA --log-traj --out $OUT/n4_delay_10 \
  && $ACT --mode te $DL $DA --log-traj --out $OUT/n4_delay_te \
  && $DP $DL 0 --out $OUT/n1_dp_delay0 \
  && $DP $DL $DD --out $OUT/n3_dp_delay ) 2>&1 | grep -v -E "$FILTER" > $OUT/chain_c.log &
PC=$!

export CHUNK_RENDER=cpu
source scripts/env.sh
export LP_NUM_THREADS=10
A1="python scripts/eval.py --config configs/act_default.yaml --k 100 --seeds 0-9 --batch-size 10 --allow-cpu-render --log-traj"
( $A1 --out $OUT/n1_act_none \
  && $A1 $NZ 0 --out $OUT/n1_act_noise0 \
  && $A1 $DL 0 --out $OUT/n1_act_delay0 ) 2>&1 | grep -v -E "$FILTER" > $OUT/chain_a.log &
PA=$!

fail=0
for p in $PA $PB $PC; do wait $p || fail=1; done
grep -ah "批 \|Error\|error" $OUT/chain_*.log | tail -40
for d in $OUT/n1_* $OUT/n3_* $OUT/n4_*; do python scripts/summarize.py $d > /dev/null || { echo "summarize 校验失败：$d"; fail=1; }; done
python scripts/t5_checks.py analyze || fail=1
exit $fail
