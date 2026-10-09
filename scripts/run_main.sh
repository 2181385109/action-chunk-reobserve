#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T6b 正式评估（耗时长，在终端手动运行）：aloha ACT（GPU 渲染）× 4 种策略 × 10 种条件 = 40 格，
#   每格 200 回合（种子 200000–200199，所有格子同一批种子），B=10。设计见 PREREGISTRATION.md。
# 用法（从 Windows，Git Bash 先 export MSYS_NO_PATHCONV=1）：
#   wsl.exe -d Ubuntu-24.04 -u root -- bash <repo>/scripts/run_main.sh [格子子集]
# 格子子集（可不写 = 全部 40 格）：
#   1-20            第 1 到第 20 格（顺序见 `python scripts/main_tools.py list`）
#   21-40           第 21 到第 40 格
#   k20__none,te__none   按格子名（逗号分隔）
# 可续跑：中断（Ctrl+C / 断电 / 关机）后重新运行同一命令即可。已完成的格子（summary.json 在且 200 个回合都在
#   episodes.csv）跳过；做到一半的格子从未完成的批次接着跑，已完成的批次不重跑。
# 种子保护：PREREGISTRATION.md 没有提交（git commit）之前，>=200000 的种子会被 seed_guard 拦住，一个回合都不会跑。
# 输出：results/main/<格子名>/（config.json、episodes.csv、summary.json）；40 格全部跑完后自动运行 scripts/analyze_main.py。
set -eo pipefail
source scripts/env.sh
MAIN_OUT="${MAIN_OUT:-results/main}"        # 只用于检查；默认值就是正式设置
SELECT="${1:-all}"
SEEDS=$(python scripts/main_tools.py seeds)
N=$(python -c "from chunkre.config import parse_seeds; print(len(parse_seeds('$SEEDS')))")

# 1) 种子保护：先于一切（加载模型之前）检查，被拦住就直接退出
if ! python -c "
from chunkre.config import parse_seeds
from chunkre.seed_guard import check_seeds
check_seeds(parse_seeds('$SEEDS'))
" 2> >(tail -n 1 >&2); then
  echo "=== seed_guard 拦截：种子 $SEEDS 在 PREREGISTRATION.md 提交之前不能使用。没有运行任何回合。" >&2
  exit 3
fi

python scripts/main_tools.py gen --out configs/main
CELLS=$(python scripts/main_tools.py list --select "$SELECT")
NC=$(echo $CELLS | wc -w)
mkdir -p "$MAIN_OUT"

echo "=== T6b 正式评估：本次 $NC 格（选择：$SELECT）× $N 回合（种子 $SEEDS，B=10），输出 $MAIN_OUT"
python scripts/main_tools.py eta --select "$SELECT" || echo "    （耗时估算失败，不影响运行）"
echo "    （中途可 Ctrl+C，重跑同一命令续跑；运行期间不要同时跑其他 GPU 任务）"

i=0
for c in $CELLS; do
  i=$((i + 1))
  if python scripts/main_tools.py done "$MAIN_OUT/$c" "$N"; then
    echo "=== [$i/$NC] $c：已完成，跳过"
    continue
  fi
  echo "=== [$i/$NC] $c  ($(date '+%H:%M:%S'))"
  python scripts/eval.py --config "configs/main/$c.yaml" --out "$MAIN_OUT/$c" 2>&1 \
    | { grep --line-buffered -v -E "UserWarning|logger.warn|task_description|deprecated|^\s*[\{\}\"]|^  " || true; } | tee -a "$MAIN_OUT.log"
done

ALL=$(python scripts/main_tools.py list)
MISSING=0
for c in $ALL; do python scripts/main_tools.py done "$MAIN_OUT/$c" "$N" || MISSING=$((MISSING + 1)); done
if [ "$MISSING" -eq 0 ]; then
  echo "=== 40 格全部完成，运行分析"
  python scripts/analyze_main.py --dir "$MAIN_OUT" --out "$MAIN_OUT"
  echo "=== 完成：$MAIN_OUT/main_report.md 和图"
else
  echo "=== 本次完成。40 格里还有 $MISSING 格没跑完（运行另一半后会自动分析）。"
fi
