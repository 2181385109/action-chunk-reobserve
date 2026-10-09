#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T11a 自适应规则阈值试跑（耗时长，在终端手动运行）：22 格 = m ∈ {10, 20} × tau ∈ 5 个候选值 × {无扰动, 推 0.06 m}
#   + k=100 在同样两种条件下的 2 格参照；种子 6000–6049（调参区间里的新种子），B=10，GPU 渲染。
#   输出 results/tune_adaptive/<格子名>/；全部完成后自动运行 scripts/select_adaptive.py（选择规则见 src/chunkre/tune_adaptive.py）
#   -> results/tune_adaptive/selection.md / selection.json。
# 用法（从 Windows，Git Bash 先 export MSYS_NO_PATHCONV=1）：
#   wsl.exe -d Ubuntu-24.04 -u root -- bash <repo>/scripts/run_tune_adaptive.sh [选择]
# 选择（可不写 = all）：all / 1-11 / 12-22（从 1 数，顺序见 `python scripts/tune_adaptive_tools.py list`）/ 格子名逗号分隔
# 可续跑：中断后重新运行同一命令即可（已完成的格子跳过；做到一半的从未完成的批次接着跑）。
# 只用于脚本自检的环境变量：TUNE_OUT（默认 results/tune_adaptive）、TUNE_CONFIGS（默认 configs/tune_adaptive）、
#   TUNE_SEEDS（默认 6000-6049）、TUNE_BATCH（默认 10）、TUNE_MAX_STEPS（默认不限）。
set -eo pipefail
source scripts/env.sh
OUT="${TUNE_OUT:-results/tune_adaptive}"
CONF="${TUNE_CONFIGS:-configs/tune_adaptive}"
SEEDS="${TUNE_SEEDS:-6000-6049}"
BATCH="${TUNE_BATCH:-10}"
SELECT="${1:-all}"
TT="python scripts/tune_adaptive_tools.py"
N=$($TT n --seeds "$SEEDS")
MS=""
[ -n "$TUNE_MAX_STEPS" ] && MS="--max-steps $TUNE_MAX_STEPS"

$TT gen --out "$CONF" --seeds "$SEEDS" --batch "$BATCH" $MS
CELLS=$($TT list --select "$SELECT")
NC=$(echo $CELLS | wc -w)
mkdir -p "$OUT"
echo "=== T11a 自适应阈值试跑：选择 $SELECT，本次 $NC 格 × $N 回合，种子 $SEEDS，B=$BATCH，输出 $OUT"
$TT eta --select "$SELECT" --seeds "$SEEDS" || echo "    （耗时估算失败，不影响运行）"
echo "    （中途可 Ctrl+C，重跑同一命令续跑；笔记本请插电、关睡眠；运行期间不要同时跑其他 GPU 任务）"

filter() { { grep --line-buffered -v -E "UserWarning|logger.warn|task_description|deprecated|^\s*[\{\}\"]|^  " || true; }; }

i=0
for c in $CELLS; do
  i=$((i + 1))
  if $TT done "$OUT/$c" "$N"; then
    echo "=== [$i/$NC] $c：已完成，跳过"
    continue
  fi
  echo "=== [$i/$NC] $c  ($(date '+%H:%M:%S'))"
  python scripts/eval.py --config "$CONF/$c.yaml" --out "$OUT/$c" 2>&1 | filter | tee -a "$OUT.log"
done

MISSING=0
for c in $($TT list); do $TT done "$OUT/$c" "$N" || MISSING=$((MISSING + 1)); done
if [ "$MISSING" -eq 0 ]; then
  echo "=== 22 格全部完成，运行选择规则"
  python scripts/select_adaptive.py --dir "$OUT" --n "$N"
else
  echo "=== 本次完成。22 格里还有 $MISSING 格没跑完（跑完剩下的以后会自动运行选择规则）。"
fi
