#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T13 失败案例分析数据（交给用户在终端跑，约 40 分钟）：预训练 ACT，7 格 = k20、k100 × {无扰动, 推 0.06, 延迟 10} + k50 延迟 10；
#   种子 7000–7049（调试区间，之前没用过），每格 50 回合，B=10，GPU 渲染，全部 --log-traj（含 T13 指尖位置），
#   每格前 3 个种子存画面（推的两格前 10 个），每 3 步一张。输出 results/failure/<格子名>/；全部完成后自动运行
#   scripts/analyze_failure.py report -> results/failure/failure_report.md + 图。
# 用法（从 Windows）：
#   wsl.exe -d Ubuntu-24.04 -u root --cd /mnt/d/xiangmu/chunk-reobserve -- bash scripts/run_failure.sh
# 可续跑：中断后重新运行同一命令即可（已完成的格子跳过；做到一半的从未完成的批次接着跑）。
# 只用于冒烟自检的环境变量：FAIL_OUT（默认 results/failure）、FAIL_SEEDS（默认 7000-7049）、FAIL_CELLS（逗号分隔，默认 7 格）、
#   FAIL_BATCH（默认 10）、FAIL_LABEL（报告第一行的标注）。
set -eo pipefail
source scripts/env.sh
OUT="${FAIL_OUT:-results/failure}"
SEEDS="${FAIL_SEEDS:-7000-7049}"
BATCH="${FAIL_BATCH:-10}"
FT="python scripts/failure_tools.py"
CELLS=$($FT list --cells "${FAIL_CELLS:-}")
N=$(python -c "import sys; sys.path.insert(0, 'src'); from chunkre.config import parse_seeds; print(len(parse_seeds('$SEEDS')))")
NC=$(echo $CELLS | wc -w)
mkdir -p "$OUT"
echo "=== T13 失败案例数据：$NC 格 × $N 回合，种子 $SEEDS，B=$BATCH，输出 $OUT"
$FT eta "$OUT" --seeds "$SEEDS" --cells "${FAIL_CELLS:-}" || echo "    （耗时估算失败，不影响运行）"
echo "    （中途可 Ctrl+C，重跑同一命令续跑；笔记本请插电、关睡眠；运行期间不要同时跑其他 GPU 任务）"

filter() { { grep --line-buffered -v -E "UserWarning|logger.warn|task_description|deprecated|^\s*[\{\}\"]|^  " || true; }; }

i=0
for c in $CELLS; do
  i=$((i + 1))
  if $FT done "$OUT/$c" "$N"; then
    echo "=== [$i/$NC] $c：已完成，跳过"
    continue
  fi
  echo "=== [$i/$NC] $c  ($(date '+%H:%M:%S'))"
  python scripts/eval.py --config configs/act_default.yaml --out "$OUT/$c" --name "$c" --seeds "$SEEDS" --batch-size "$BATCH" \
    --log-traj $($FT args "$c" --seeds "$SEEDS") 2>&1 | filter | tee -a "$OUT.log"
done

MISSING=0
for c in $CELLS; do $FT done "$OUT/$c" "$N" || MISSING=$((MISSING + 1)); done
if [ "$MISSING" -eq 0 ]; then
  echo "=== 全部完成，生成报告"
  if [ -n "$FAIL_LABEL" ]; then
    python scripts/analyze_failure.py report --dir "$OUT" --label "$FAIL_LABEL"
  else
    python scripts/analyze_failure.py report --dir "$OUT"
  fi
  du -sh "$OUT" 2>/dev/null || true
  du -sh "$OUT"/*/frames 2>/dev/null | tail -n 20 || true
else
  echo "=== 还有 $MISSING 格没跑完，重跑同一命令续跑"
fi
