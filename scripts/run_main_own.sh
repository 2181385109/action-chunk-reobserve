#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T10b 自训练 ACT 正式评估（交给用户在终端跑）：先跑基础水平测试（k=100，种子 1000–1499，500 回合，输出 results/own_baseline/），
#   再跑 9 格（k ∈ {20, 50, 100} × {无扰动；推方块 0.06 m；动作延迟 10 步}，每格 200 回合，种子 200000–200199，
#   所有格子同一批种子，B=10，GPU 渲染），输出 results/main_own/<格子名>/。设计见 PREREGISTRATION_OWN.md。
#   模型 = 自训练 ACT 的第 100000 步（最终）检查点，路径和哈希写死（src/chunkre/main_own.py），config.json 里记录路径和文件哈希。
# 用法（从 Windows，Git Bash 先 export MSYS_NO_PATHCONV=1）：
#   wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/d/xiangmu/chunk-reobserve/scripts/run_main_own.sh [选择]
# 选择（可不写 = all）：
#   all             基础水平测试 + 9 格（顺序：基础水平测试，再 `python scripts/main_own_tools.py list` 的 9 格）
#   baseline        只跑基础水平测试
#   cells           只跑 9 格（不含基础水平测试）
#   1-5   6-9       9 格里的第 1 到第 5 格 / 第 6 到第 9 格（不含基础水平测试）
#   k20__none,k50__none   按格子名（逗号分隔）
# 可续跑：中断（Ctrl+C / 断电 / 关机）后重新运行同一命令即可。已完成的（summary.json 在且回合都在 episodes.csv）跳过；
#   做到一半的从未完成的批次接着跑，已完成的批次不重跑（每批 10 回合）。可以分批：先 `baseline`，再 `1-5`、`6-9`。
# 登记保护：PREREGISTRATION_OWN.md 没有提交（git commit）之前，一个回合都不会跑（含基础水平测试）；
#   检查点的 SHA-256 与登记不一致也不跑。
# 全部完成（基础水平测试 + 9 格）后自动运行 scripts/analyze_main_own.py -> results/main_own/main_own_report.md 等。
# 可选环境变量（只用于检查）：MAIN_OWN_OUT（默认 results/main_own）、OWN_BASELINE_OUT（默认 results/own_baseline）。
set -eo pipefail
source /mnt/d/xiangmu/chunk-reobserve/scripts/env.sh
MAIN_OUT="${MAIN_OWN_OUT:-results/main_own}"
BASE_OUT="${OWN_BASELINE_OUT:-results/own_baseline}"
SELECT="${1:-all}"
MT="python scripts/main_own_tools.py"
SEEDS=$($MT seeds)
N=$(python -c "from chunkre.config import parse_seeds; print(len(parse_seeds('$SEEDS')))")
BASE_N=500   # 种子 1000–1499

# 1) 登记保护：先于一切（加载模型之前）检查，被拦住就直接退出
if ! $MT guard 2> >(tail -n 1 >&2); then
  echo "=== 登记保护拦截：PREREGISTRATION_OWN.md 提交之前不能评估自训练模型。没有运行任何回合。" >&2
  exit 3
fi
# 2) 检查点必须是登记的那一个
if ! $MT verify-ckpt; then
  echo "=== 检查点核对失败：不是登记的第 100000 步检查点（路径或哈希不符）。没有运行任何回合。" >&2
  exit 4
fi

DO_BASE=0
case "$SELECT" in
  all) DO_BASE=1; CELL_SEL=all ;;
  baseline) DO_BASE=1; CELL_SEL=none ;;
  cells) CELL_SEL=all ;;
  *) CELL_SEL="$SELECT" ;;
esac

$MT gen --out configs/main_own
if [ "$CELL_SEL" = none ]; then CELLS=""; else CELLS=$($MT list --select "$CELL_SEL"); fi
NC=$(echo $CELLS | wc -w)
mkdir -p "$MAIN_OUT"

echo "=== T10b 自训练 ACT 正式评估：选择 $SELECT（基础水平测试 $([ $DO_BASE = 1 ] && echo 是 || echo 否)，9 格里本次 $NC 格 × $N 回合，种子 $SEEDS，B=10）"
echo "    检查点 $($MT ckpt)（哈希已核对）；输出 $BASE_OUT 与 $MAIN_OUT"
if [ $DO_BASE = 1 ]; then BFLAG=""; else BFLAG="--no-baseline"; fi
$MT eta --select "$CELL_SEL" $BFLAG || echo "    （耗时估算失败，不影响运行）"
echo "    （中途可 Ctrl+C，重跑同一命令续跑；笔记本请插电、关睡眠；运行期间不要同时跑其他 GPU 任务）"

filter() { { grep --line-buffered -v -E "UserWarning|logger.warn|task_description|deprecated|^\s*[\{\}\"]|^  " || true; }; }

if [ $DO_BASE = 1 ]; then
  if $MT done "$BASE_OUT" "$BASE_N"; then
    echo "=== [基础水平测试] 已完成，跳过"
  else
    echo "=== [基础水平测试] k=100，种子 1000–1499，500 回合  ($(date '+%H:%M:%S'))"
    python scripts/eval.py --config configs/main_own/own_baseline.yaml --out "$BASE_OUT" 2>&1 | filter | tee -a "$BASE_OUT.log"
  fi
fi

i=0
for c in $CELLS; do
  i=$((i + 1))
  if $MT done "$MAIN_OUT/$c" "$N"; then
    echo "=== [$i/$NC] $c：已完成，跳过"
    continue
  fi
  echo "=== [$i/$NC] $c  ($(date '+%H:%M:%S'))"
  python scripts/eval.py --config "configs/main_own/$c.yaml" --out "$MAIN_OUT/$c" 2>&1 | filter | tee -a "$MAIN_OUT.log"
done

ALL=$($MT list)
NALL=$(echo $ALL | wc -w)
MISSING=0
for c in $ALL; do $MT done "$MAIN_OUT/$c" "$N" || MISSING=$((MISSING + 1)); done
$MT done "$BASE_OUT" "$BASE_N" || MISSING=$((MISSING + 1))
if [ "$MISSING" -eq 0 ]; then
  echo "=== 基础水平测试 + $NALL 格全部完成，运行分析"
  python scripts/analyze_main_own.py --dir "$MAIN_OUT" --baseline-dir "$BASE_OUT" --out "$MAIN_OUT"
  echo "=== 完成：$MAIN_OUT/main_own_report.md 和图"
else
  echo "=== 本次完成。基础水平测试 + $NALL 格里还有 $MISSING 项没跑完（跑完剩下的以后会自动分析）。"
fi
