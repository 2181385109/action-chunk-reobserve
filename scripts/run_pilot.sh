#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T6a 试跑（交给用户在终端跑；执行对话不自己跑完整试跑）。
#   aloha ACT（GPU 渲染）：5 种策略（k=10/20/50/100 + 时间集成）× 11 种扰动条件 = 55 格，
#   每格种子 5000–5019（20 回合），B=10。只用于调参（定强度档位和每格回合数），不作为结论。
# 从 Windows（Git Bash 先 export MSYS_NO_PATHCONV=1）：
#   wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/d/xiangmu/chunk-reobserve/scripts/run_pilot.sh
# 中断后重新运行同一命令即可续跑：已完成的格子（summary.json 在且 20 个回合都在 episodes.csv）跳过，
#   做到一半的格子从未完成的批次接着跑。结果：results/pilot/<格子名>/；全部跑完后自动运行 scripts/analyze_pilot.py。
# 可选环境变量（只用于试跑 / 检查）：PILOT_OUT（默认 results/pilot）、PILOT_CONFIGS（默认 configs/pilot）、
#   PILOT_SEEDS（默认 5000-5019）、PILOT_B（默认 10）、PILOT_CELLS（逗号分隔的格子名，默认全部 55 个）。
#   设了 PILOT_SEEDS / PILOT_B 时配置会重新生成到 PILOT_CONFIGS（请同时设一个临时的 PILOT_CONFIGS 和 PILOT_OUT）。
set -eo pipefail
source /mnt/d/xiangmu/chunk-reobserve/scripts/env.sh
PILOT_OUT="${PILOT_OUT:-results/pilot}"
PILOT_CONFIGS="${PILOT_CONFIGS:-configs/pilot}"
PILOT_SEEDS="${PILOT_SEEDS:-5000-5019}"
PILOT_B="${PILOT_B:-10}"
N=$(python -c "from chunkre.config import parse_seeds; print(len(parse_seeds('$PILOT_SEEDS')))")

# 配置每次按当前种子 / B 重新生成（幂等；文件内容与已提交的 configs/pilot 一致时不产生改动）
python scripts/pilot_tools.py gen --out "$PILOT_CONFIGS" --seeds "$PILOT_SEEDS" --batch-size "$PILOT_B"
if [ -n "$PILOT_CELLS" ]; then CELLS="${PILOT_CELLS//,/ }"; else CELLS=$(python scripts/pilot_tools.py list); fi
NC=$(echo $CELLS | wc -w)
mkdir -p "$PILOT_OUT"

echo "=== T6a 试跑：$NC 格 × $N 回合（种子 $PILOT_SEEDS，B=$PILOT_B），输出 $PILOT_OUT"
CELLS_CSV=$(echo $CELLS | tr ' ' ',')
python scripts/pilot_tools.py eta --n "$N" --B "$PILOT_B" --cells "$CELLS_CSV" || echo "    （耗时估算失败，不影响运行）"
echo "    （中途可 Ctrl+C，重跑同一命令续跑）"

i=0
for c in $CELLS; do
  i=$((i + 1))
  if python scripts/pilot_tools.py done "$PILOT_OUT/$c" "$N"; then
    echo "=== [$i/$NC] $c：已完成，跳过"
    continue
  fi
  echo "=== [$i/$NC] $c  ($(date '+%H:%M:%S'))"
  python scripts/eval.py --config "$PILOT_CONFIGS/$c.yaml" --out "$PILOT_OUT/$c" 2>&1 \
    | { grep --line-buffered -v -E "UserWarning|logger.warn|task_description|deprecated|^\s*[\{\}\"]|^  " || true; } | tee -a "$PILOT_OUT.log"
done

echo "=== 汇总"
python scripts/analyze_pilot.py --dir "$PILOT_OUT" --expected-n "$N"
echo "=== 完成：$PILOT_OUT/pilot_table.md、pilot_table.csv、fig_*.png"
