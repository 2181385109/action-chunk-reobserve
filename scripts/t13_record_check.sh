#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T13：指尖记录前后动作逐位相同（CPU llvmpipe 渲染，种子 0–4，B=5，约 10 分钟）。
# 旧代码 = OLD_COMMIT（加指尖记录之前的提交）用 git archive 解到 /tmp/t13_old；新代码 = 当前仓库。
# 条件：k20 无扰动 / k20 推 0.06 / k20 延迟 10（覆盖改状态、改动作两个钩子）。结果 results/failure_checks/record_check.md
set -eo pipefail
OLD_COMMIT="${OLD_COMMIT:-e970cfd}"
export CHUNK_RENDER=cpu
source scripts/env.sh
W=/tmp/t13_check
rm -rf "$W" /tmp/t13_old
mkdir -p "$W" /tmp/t13_old
git archive "$OLD_COMMIT" | tar -x -C /tmp/t13_old
filter() { { grep --line-buffered -v -E "UserWarning|logger.warn|task_description|deprecated|^\s*[\{\}\"]|^  " || true; }; }
run() {   # 代码根 输出 名字 参数...
  local root=$1 out=$2 name=$3; shift 3
  python "$root/scripts/eval.py" --config configs/act_default.yaml --out "$out/$name" --k 20 --seeds 0-4 --batch-size 5 \
    --allow-cpu-render --log-traj "$@" 2>&1 | filter
}
PUSH="--perturb push --perturb-strength 0.06 --perturb-param t_lo=20 --perturb-param t_hi=100"
DELAY="--perturb action_delay --perturb-strength 10"
for spec in "none|" "push0.06|$PUSH" "delay10|$DELAY"; do
  name="k20__${spec%%|*}"; args="${spec#*|}"
  run /tmp/t13_old "$W/old" "$name" $args &
  run "$PWD" "$W/new" "$name" $args &
  wait
done
python scripts/t13_record_check.py compare "$W/old" "$W/new" k20__none k20__push0.06 k20__delay10
