#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T2 排查 ACT 不确定性：渲染画面 / 策略推理 是否跨进程逐字节可复现。结果写 results/t2_checks/probe_determinism.jsonl
source scripts/env.sh
OUT=results/t2_checks/probe_determinism.jsonl
: > $OUT
P="python scripts/probe_determinism.py"
run() { "$@" 2>/dev/null | grep '^PROBE ' | sed 's/^PROBE //' >> $OUT; }
STEPS="${STEPS:-60}"
run $P pusht_order --tag pusht_order
run $P act_infer --B 10 --tag infer_a
run $P act_infer --B 10 --tag infer_b
run $P render --B 1 --steps $STEPS --tag gpu_B1_a
run $P render --B 1 --steps $STEPS --tag gpu_B1_b
run $P render --B 10 --steps $STEPS --tag gpu_B10_a
run $P render --B 10 --steps $STEPS --tag gpu_B10_b
# 并发 GPU 负载下
$P gpu_load --seconds 150 > /dev/null 2>&1 &
LP=$!
sleep 5
run $P render --B 10 --steps $STEPS --tag gpu_B10_load
run $P act_infer --B 10 --tag infer_load
kill $LP 2>/dev/null; wait $LP 2>/dev/null
# CPU 软渲染对照
if [ "${WITH_CPU:-1}" = "1" ]; then
  export CHUNK_RENDER=cpu
  source scripts/env.sh
  run $P render --B 10 --steps $STEPS --tag cpu_B10_a
  run $P render --B 10 --steps $STEPS --tag cpu_B10_b
fi
python - <<'PY'
import json
rs = [json.loads(l) for l in open("results/t2_checks/probe_determinism.jsonl")]
for r in rs:
    if r["mode"] == "act_infer":
        print(r["tag"], "hashes", r["hashes"])
    elif r["mode"] == "pusht_order":
        print(r)
    else:
        print(r["tag"], "digest", r["digest"], "renderers", r["renderers"])
def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a["per_step"], b["per_step"])):
        if x != y:
            return i
    return None
by = {r["tag"]: r for r in rs}
for a, b in [("gpu_B1_a", "gpu_B1_b"), ("gpu_B10_a", "gpu_B10_b"), ("gpu_B10_a", "gpu_B10_load"), ("cpu_B10_a", "cpu_B10_b")]:
    if a in by and b in by:
        print(f"{a} vs {b}: 第一个画面不同的步 = {first_diff(by[a], by[b])}，最终 qpos 相同 = {by[a]['final_qpos'] == by[b]['final_qpos']}")
PY
