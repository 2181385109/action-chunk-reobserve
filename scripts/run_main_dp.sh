#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T9 DP 正式评估（耗时长，在终端手动运行）：PushT Diffusion Policy（lerobot/diffusion_pusht）× 3 种执行长度 k ∈ {4, 8, 15}
#   × 4 种条件（无扰动；推 T 形块 d=60；动作延迟 2 步；动作延迟 4 步）= 12 格，
#   每格 200 回合（种子 200000–200199，所有格子同一批种子），B=50。设计见 PREREGISTRATION_DP.md。
# 用法（从 Windows，Git Bash 先 export MSYS_NO_PATHCONV=1）：
#   wsl.exe -d Ubuntu-24.04 -u root -- bash <repo>/scripts/run_main_dp.sh [格子子集]
# 格子子集（可不写 = 全部 12 格）：
#   1-6             第 1 到第 6 格（顺序见 `python scripts/main_dp_tools.py list`）
#   7-12            第 7 到第 12 格
#   k4__none,k8__none   按格子名（逗号分隔）
# 可续跑：中断（Ctrl+C / 断电 / 关机）后重新运行同一命令即可。已完成的格子（summary.json 在且 200 个回合都在
#   episodes.csv）跳过；做到一半的格子从未完成的批次接着跑，已完成的批次不重跑（每批 50 回合，每格 4 批）。
# 种子保护：PREREGISTRATION_DP.md 没有提交（git commit）之前，>=200000 的种子会被 seed_guard 拦住，一个回合都不会跑。
# 输出：results/main_dp/<格子名>/（config.json、episodes.csv、summary.json）；12 格全部跑完后自动运行
#   scripts/analyze_main_dp.py（H4–H5 + 次要指标）和 scripts/analyze_interaction.py（探索性“差距变化”，要读 results/main/）。
# 可选环境变量（只用于检查）：MAIN_DP_OUT（默认 results/main_dp）。
set -eo pipefail
source scripts/env.sh
MAIN_OUT="${MAIN_DP_OUT:-results/main_dp}"
SELECT="${1:-all}"
MT="python scripts/main_dp_tools.py"
SEEDS=$($MT seeds)
N=$(python -c "from chunkre.config import parse_seeds; print(len(parse_seeds('$SEEDS')))")

# 1) 种子保护：先于一切（加载模型之前）检查，被拦住就直接退出。model="dp" -> 看 PREREGISTRATION_DP.md
if ! python -c "
from chunkre.config import parse_seeds
from chunkre.seed_guard import check_seeds
check_seeds(parse_seeds('$SEEDS'), model='dp')
" 2> >(tail -n 1 >&2); then
  echo "=== seed_guard 拦截：种子 $SEEDS 在 PREREGISTRATION_DP.md 提交之前不能使用。没有运行任何回合。" >&2
  exit 3
fi

$MT gen --out configs/main_dp
CELLS=$($MT list --select "$SELECT")
NC=$(echo $CELLS | wc -w)
mkdir -p "$MAIN_OUT"

echo "=== T9 DP 正式评估：本次 $NC 格（选择：$SELECT）× $N 回合（种子 $SEEDS，B=50），输出 $MAIN_OUT"
$MT eta --select "$SELECT" || echo "    （耗时估算失败，不影响运行）"
echo "    （中途可 Ctrl+C，重跑同一命令续跑；运行期间不要同时跑其他 GPU 任务）"

i=0
for c in $CELLS; do
  i=$((i + 1))
  if $MT done "$MAIN_OUT/$c" "$N"; then
    echo "=== [$i/$NC] $c：已完成，跳过"
    continue
  fi
  echo "=== [$i/$NC] $c  ($(date '+%H:%M:%S'))"
  python scripts/eval.py --config "configs/main_dp/$c.yaml" --out "$MAIN_OUT/$c" 2>&1 \
    | { grep --line-buffered -v -E "UserWarning|logger.warn|task_description|deprecated|^\s*[\{\}\"]|^  " || true; } | tee -a "$MAIN_OUT.log"
done

ALL=$($MT list)
NALL=$(echo $ALL | wc -w)
MISSING=0
for c in $ALL; do $MT done "$MAIN_OUT/$c" "$N" || MISSING=$((MISSING + 1)); done
if [ "$MISSING" -eq 0 ]; then
  echo "=== $NALL 格全部完成，运行分析"
  python scripts/analyze_main_dp.py --dir "$MAIN_OUT" --out "$MAIN_OUT"
  python scripts/analyze_interaction.py --act-dir results/main --dp-dir "$MAIN_OUT" --out "$MAIN_OUT/interaction"
  echo "=== 完成：$MAIN_OUT/main_dp_report.md、$MAIN_OUT/interaction/interaction_report.md 和图"
else
  echo "=== 本次完成。$NALL 格里还有 $MISSING 格没跑完（跑完剩下的格子后会自动分析）。"
fi
