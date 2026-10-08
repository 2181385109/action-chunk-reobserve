#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T10a：演示 train_act.sh 中断后续训（约 3 分钟）。小规模：300 步、每 100 步存检查点、每 20 步记日志。
# 第 1 次：跑到第 200 步检查点之后几秒，用 SIGKILL 杀掉整个进程组（模拟断电 / WSL 被关）；
# 第 2 次：同一命令重跑 → 应从第 200 步续训，跑完 300 步。结果 -> results/train_act_own/resume_demo/
source scripts/env.sh
export STEPS=300 SAVE_FREQ=100 LOG_FREQ=20 WORKERS=4
export OUT=/root/train_out/act_resume_demo LOG_DIR=results/train_act_own/resume_demo
rm -rf "$OUT" "$LOG_DIR"
mkdir -p "$LOG_DIR"

echo "===== 第 1 次运行（将被杀掉）====="
setsid bash scripts/train_act.sh > "$LOG_DIR/run1_console.txt" 2>&1 &
PG=$!
for _ in $(seq 1 600); do
  if grep -q "Checkpoint policy after step 200" "$LOG_DIR/train_log.txt" 2>/dev/null && [ -L "$OUT/checkpoints/last" ] \
     && [ "$(readlink "$OUT/checkpoints/last")" = "000200" ]; then
    break
  fi
  sleep 1
done
sleep 6                                   # 再多跑几十步，让中断发生在两个检查点之间
KILLED_AT=$(grep -c " step:" "$LOG_DIR/train_log.txt")
kill -KILL -- "-$PG"
wait "$PG" 2>/dev/null
echo "杀掉时日志里有 $KILLED_AT 行 step 记录；最新检查点 -> $(readlink "$OUT/checkpoints/last")"
ls "$OUT/checkpoints"

echo "===== 第 2 次运行（同一命令，应续训）====="
bash scripts/train_act.sh > "$LOG_DIR/run2_console.txt" 2>&1
RC=$?
grep -E "\[train_act\]" "$LOG_DIR/run2_console.txt"
echo "第 2 次退出码 $RC"
ls "$OUT/checkpoints"
python - "$LOG_DIR" "$OUT" "$KILLED_AT" "$RC" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from train_act_tools import RESUME_MARK, checkpoints_in_log, parse_train_log
d, out, killed_at, rc = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
text = (d / "train_log.txt").read_text(encoding="utf-8")
recs = parse_train_log(text, log_freq=20)
marks = [l for l in text.splitlines() if l.startswith(RESUME_MARK)]
ts = json.loads((out / "checkpoints" / "last" / "training_state" / "training_step.json").read_text())
res = dict(
    run2_returncode=rc, start_marks=marks, step_lines_logged_before_kill=killed_at,
    checkpoints_logged=checkpoints_in_log(text), checkpoint_dirs=sorted(p.name for p in (out / "checkpoints").iterdir()),
    final_training_step=ts["step"], curve_steps=[r["step"] for r in recs], curve_runs=[r["run"] for r in recs],
    losses=[r["loss"] for r in recs],
    resumed_from_200=(len(marks) == 2 and marks[1].split()[2:4] == ["step=200", "mode=resume"]),
    curve_complete=[r["step"] for r in recs] == list(range(20, 301, 20)),
)
res["pass"] = bool(rc == 0 and res["resumed_from_200"] and res["curve_complete"] and ts["step"] == 300
                   and killed_at * 20 > 200)
(d / "demo_summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "losses"}, ensure_ascii=False))
EOF
