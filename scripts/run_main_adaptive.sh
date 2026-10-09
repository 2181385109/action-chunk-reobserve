#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T11b 自适应规则正式评估（耗时长，在终端手动运行）：2 格 ad__none、ad__push0.06（自适应规则 m=10、tau=0.2906、h_cmp=20、min_gap=20；
#   种子 200000–200199，每格 200 回合，同一批种子，B=10，GPU 渲染），输出 results/main_adaptive/<格子名>/。设计见 PREREGISTRATION_ADAPTIVE.md。
#   对照 = results/main/k100__none、k100__push0.06（同一批种子，不重跑）。
# 用法（从 Windows，Git Bash 先 export MSYS_NO_PATHCONV=1）：
#   wsl.exe -d Ubuntu-24.04 -u root --cd <repo> -- bash scripts/run_main_adaptive.sh [选择]
# 选择（可不写 = all）：all | 1 | 2 | ad__none,ad__push0.06（按格子名，逗号分隔）
# 可续跑：中断后重新运行同一命令即可。已完成的（summary.json 在且回合都在 episodes.csv）跳过；做到一半的从未完成的批次接着跑。
# 开跑前的两道检查（任何一道不过，一个回合都不跑、不加载模型）：
#   1) PREREGISTRATION_ADAPTIVE.md 必须已 git 提交（否则 exit 3）；
#   2) 要跑的配置里的 m、tau、h_cmp、min_gap 必须与登记文件里写的数字逐项相同（否则 exit 4）。
# 全部完成后自动运行 scripts/analyze_main_adaptive.py -> results/main_adaptive/main_adaptive_report.md 等。
# 可选环境变量（只用于检查 / 演示，正式运行不要设）：MAIN_ADAPTIVE_OUT（默认 results/main_adaptive）、
#   MAIN_ADAPTIVE_CONFIG_DIR（默认 configs/main_adaptive）、ADAPT_M / ADAPT_TAU / ADAPT_H_CMP / ADAPT_MIN_GAP（覆盖参数：演示“参数不一致被拦”）。
set -eo pipefail
source scripts/env.sh
OUT="${MAIN_ADAPTIVE_OUT:-results/main_adaptive}"
CFG="${MAIN_ADAPTIVE_CONFIG_DIR:-configs/main_adaptive}"
SELECT="${1:-all}"
MT="python scripts/main_adaptive_tools.py"
SEEDS=$($MT seeds)
N=$(python -c "from chunkre.config import parse_seeds; print(len(parse_seeds('$SEEDS')))")

# 1) 登记保护：先于一切（加载模型之前）
if ! $MT guard 2> >(tail -n 1 >&2); then
  echo "=== 登记保护拦截：PREREGISTRATION_ADAPTIVE.md 提交之前不能用最终种子评估自适应规则。没有运行任何回合。" >&2
  exit 3
fi
# 2) 配置生成 + 参数与登记逐项核对
GEN_ARGS=""
[ -n "$ADAPT_M" ] && GEN_ARGS="$GEN_ARGS --m $ADAPT_M"
[ -n "$ADAPT_TAU" ] && GEN_ARGS="$GEN_ARGS --tau $ADAPT_TAU"
[ -n "$ADAPT_H_CMP" ] && GEN_ARGS="$GEN_ARGS --h-cmp $ADAPT_H_CMP"
[ -n "$ADAPT_MIN_GAP" ] && GEN_ARGS="$GEN_ARGS --min-gap $ADAPT_MIN_GAP"
$MT gen --out "$CFG" $GEN_ARGS
if ! $MT verify-params --out "$CFG"; then
  echo "=== 参数核对失败：要跑的自适应参数与 PREREGISTRATION_ADAPTIVE.md 不一致。没有运行任何回合。" >&2
  exit 4
fi

ALL=$($MT list)
case "$SELECT" in
  all) CELLS="$ALL" ;;
  1) CELLS=$(echo "$ALL" | sed -n 1p) ;;
  2) CELLS=$(echo "$ALL" | sed -n 2p) ;;
  *) CELLS=$(echo "$SELECT" | tr ',' ' ') ;;
esac
for c in $CELLS; do
  echo "$ALL" | grep -qx "$c" || { echo "未知格子名：$c" >&2; exit 2; }
done
NC=$(echo $CELLS | wc -w)
mkdir -p "$OUT"

echo "=== T11b 自适应规则正式评估：选择 $SELECT（$NC 格 × $N 回合，种子 $SEEDS，B=10）；输出 $OUT"
echo "    参数 $($MT params)（与登记逐项核对通过）"
$MT eta --select "$(echo $CELLS | tr ' ' ',')" || echo "    （耗时估算失败，不影响运行）"
echo "    （中途可 Ctrl+C，重跑同一命令续跑；笔记本请插电、关睡眠；运行期间不要同时跑其他 GPU 任务）"

filter() { { grep --line-buffered -v -E "UserWarning|logger.warn|task_description|deprecated|^\s*[\{\}\"]|^  " || true; }; }

i=0
for c in $CELLS; do
  i=$((i + 1))
  if $MT done "$OUT/$c" "$N"; then
    echo "=== [$i/$NC] $c：已完成，跳过"
    continue
  fi
  echo "=== [$i/$NC] $c  ($(date '+%H:%M:%S'))"
  python scripts/eval.py --config "$CFG/$c.yaml" --out "$OUT/$c" 2>&1 | filter | tee -a "$OUT.log"
done

MISSING=0
for c in $ALL; do $MT done "$OUT/$c" "$N" || MISSING=$((MISSING + 1)); done
if [ "$MISSING" -eq 0 ]; then
  echo "=== 2 格全部完成，运行分析"
  python scripts/analyze_main_adaptive.py --dir "$OUT" --out "$OUT"
  echo "=== 完成：$OUT/main_adaptive_report.md 和图"
else
  echo "=== 本次完成。2 格里还有 $MISSING 格没跑完（跑完剩下的以后会自动分析）。"
fi
