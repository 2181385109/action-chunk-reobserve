#!/bin/bash
# T2 正确性检查 V1–V4。V5 耗时基准另见 scripts/bench_timing.sh（必须单独跑）。
# 用法：bash scripts/t2_checks.sh                  （默认 ACT、DP 依次串行；先删掉该模型旧的 results/t2_checks/<model>_*）
#       MODELS=act bash scripts/t2_checks.sh       （只重跑 ACT）
#       PARALLEL=1 bash scripts/t2_checks.sh       （两条链并行；注意：并行时 GPU 被另一个进程占用，见 HANDOFF“ACT 不确定性”）
set -eo pipefail
source /mnt/d/xiangmu/chunk-reobserve/scripts/env.sh
OUT=results/t2_checks
mkdir -p $OUT
MODELS="${MODELS:-act dp}"

chain() {  # $1 = act / dp
  python scripts/eval.py --config configs/$1_default.yaml --out $OUT/$1_v1_run1 --seeds 0-9 --batch-size 10
  python scripts/t2_checks.py run $1
}
fail=0
pids=()
for m in $MODELS; do
  if [ "${FRESH:-1}" = "1" ]; then rm -rf $OUT/${m}_v*; fi
  if [ "${PARALLEL:-0}" = "1" ]; then
    chain $m > $OUT/${m}_chain.log 2>&1 &
    pids+=($!)
  else
    chain $m > $OUT/${m}_chain.log 2>&1 || { echo "$m 链失败，见 $OUT/${m}_chain.log"; fail=1; }
  fi
done
for p in "${pids[@]}"; do wait $p || { echo "有一条链失败，见 $OUT/*_chain.log"; fail=1; }; done
for m in $MODELS; do grep -ah "批 \|V3" $OUT/${m}_chain.log || true; done
python scripts/t2_checks.py analyze
exit $fail
