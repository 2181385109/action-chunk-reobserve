#!/bin/bash
# 官方数字复现（交给用户在终端跑；执行对话不自己跑）。
#   ACT k=100（默认）与 DP k=8（默认），各跑官方评估种子 1000–1499 共 500 回合，然后与官方 83.0% / 65.4% 比较。
# 从 Windows（Git Bash 先 export MSYS_NO_PATHCONV=1）：
#   wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/d/xiangmu/chunk-reobserve/scripts/run_repro.sh
# 中断后重新运行同一命令即可续跑（已写入 CSV 的批次跳过）。
# 可选环境变量（只用于试跑）：REPRO_OUT（默认 results/repro）、SEEDS（默认 1000-1499）、
#   B_ACT / B_DP（默认见下，由 V5 耗时基准选出）、MODELS（默认 "act dp"）。
set -eo pipefail
source /mnt/d/xiangmu/chunk-reobserve/scripts/env.sh
REPRO_OUT="${REPRO_OUT:-results/repro}"
SEEDS="${SEEDS:-1000-1499}"
B_ACT="${B_ACT:-10}"   # V5：ACT B=10 与 25 每回合同为约 5.6 s，B=50 环境步进超线性变慢；取 10（续跑粒度更细）
B_DP="${B_DP:-50}"    # V5：DP B=50 最快（约 2.35 s/回合）
MODELS="${MODELS:-act dp}"
mkdir -p "$REPRO_OUT"
N=$(python -c "from chunkre.config import parse_seeds; print(len(parse_seeds('$SEEDS')))")

echo "=== 官方数字复现：种子 $SEEDS（$N 回合/模型），输出 $REPRO_OUT"
echo "=== 预计耗时（按 V5 实测估算，偏保守）："
for m in $MODELS; do
  if [ "$m" = act ]; then B=$B_ACT; K=100; else B=$B_DP; K=8; fi
  python scripts/bench_timing.py eta $m --k $K --B $B --n $N
done
echo "    （估算不含模型加载和建环境，约每个模型 +1 分钟；中途可 Ctrl+C，重跑同一命令续跑）"

for m in $MODELS; do
  if [ "$m" = act ]; then B=$B_ACT; K=100; else B=$B_DP; K=8; fi
  echo "=== $m: k=$K B=$B"
  python scripts/eval.py --config configs/repro_$m.yaml --out "$REPRO_OUT/$m" --seeds "$SEEDS" --batch-size $B --k $K \
    2>&1 | grep -v -E "UserWarning|logger.warn|task_description|deprecated" | tee -a "$REPRO_OUT.$m.log"
done

python scripts/summarize.py "$REPRO_OUT"/act "$REPRO_OUT"/dp 2>/dev/null | grep "summarize" || true
python scripts/compare_official.py --act "$REPRO_OUT/act" --dp "$REPRO_OUT/dp" --out "$REPRO_OUT/compare_official.json"
