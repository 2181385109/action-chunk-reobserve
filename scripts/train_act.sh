#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T10a：自训练 ACT（用户在终端跑）。lerobot 0.3.3 自带训练脚本（python -m lerobot.scripts.train，不改源码），官方超参数。
#   wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/d/xiangmu/chunk-reobserve/scripts/train_act.sh
# 中断（Ctrl+C、关机、WSL 被关）后重跑同一命令 = 从最新检查点（每 5000 步一个）续训；中断前最后一个检查点之后的步数会重跑。
# 输出（权重、检查点）在 WSL 的 $OUT（默认 /root/train_out/act_own/，不进仓库）；
# 日志在 $LOG_DIR（默认 results/train_act_own/）：train_log.txt（原始日志，多次续训追加）、loss.csv、loss.png。
# 环境变量只给测试 / 演示用（正式训练不要改）：STEPS WORKERS SAVE_FREQ LOG_FREQ OUT LOG_DIR
set -o pipefail
source scripts/env.sh

# ---- 正式设置（T10a 实测后确定，见 HANDOFF.md“训练步数怎么定的”）----
STEPS="${STEPS:-100000}"          # 官方步数；实测 5.84 步/秒 → 估计 4.76 小时 <= 10 小时 → 用官方步数
WORKERS="${WORKERS:-4}"           # 与官方相同；实测 WSL 内存峰值 3.8 / 9.7 GB
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-200}"
OUT="${OUT:-/root/train_out/act_own}"
LOG_DIR="${LOG_DIR:-results/train_act_own}"
SPS=5.84                          # 实测稳态速度（步/秒，workers=4，results/train_act_own/prep/bench_w4_s1000.json）
STARTUP_S=15                      # 实测启动到开始训练约 10 s，取整留余量

LOG="$LOG_DIR/train_log.txt"
LAST="$OUT/checkpoints/last"
mkdir -p "$LOG_DIR"

if [ -L "$LAST" ] && [ -f "$LAST/pretrained_model/train_config.json" ] && [ -f "$LAST/training_state/training_step.json" ]; then
  MODE=resume
  DONE=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["step"])' "$LAST/training_state/training_step.json")
  CFG_STEPS=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["steps"])' "$LAST/pretrained_model/train_config.json")
  if [ "$CFG_STEPS" != "$STEPS" ]; then
    echo "[train_act] 错误：$OUT 里的训练总步数是 $CFG_STEPS，与本次 STEPS=$STEPS 不同。换输出目录或改回 STEPS。" >&2
    exit 2
  fi
else
  MODE=fresh
  DONE=0
  if [ -e "$OUT" ]; then           # 上次在第一个检查点之前就中断了：没有可续的状态，挪开后从头开始（不删除）
    BAK="$OUT.incomplete_$(date +%Y%m%d_%H%M%S)"
    echo "[train_act] $OUT 存在但没有检查点，挪到 $BAK，从头开始"
    mv "$OUT" "$BAK"
  fi
fi

if [ "$DONE" -ge "$STEPS" ]; then
  echo "[train_act] 已完成：$OUT 的最新检查点是第 $DONE 步（目标 $STEPS）。只更新 loss 曲线。"
  python scripts/train_act_tools.py parse "$LOG" --out-dir "$LOG_DIR" --log-freq "$LOG_FREQ"
  exit 0
fi

ETA=$(python scripts/train_act_tools.py eta --sps "$SPS" --steps "$STEPS" --done "$DONE" --startup "$STARTUP_S")
echo "=================================================================="
echo "[train_act] 模式：$MODE（已完成 $DONE / $STEPS 步），workers=$WORKERS，种子 1000，每 $SAVE_FREQ 步存检查点"
echo "[train_act] 预计耗时：约 $ETA 小时（按实测 $SPS 步/秒；笔记本降频、同时跑其他 GPU 任务会更慢）"
echo "[train_act] 输出：$OUT    日志：$LOG"
echo "=================================================================="
echo "### TRAIN_ACT_START step=$DONE mode=$MODE time=$(date -Iseconds) steps=$STEPS workers=$WORKERS git=$(git rev-parse --short HEAD 2>/dev/null)" >> "$LOG"

if [ "$MODE" = resume ]; then
  python -m lerobot.scripts.train --config_path="$LAST/pretrained_model/train_config.json" --resume=true 2>&1 \
    | grep --line-buffered -v -E "UserWarning|warnings.warn" | tee -a "$LOG"
else
  mapfile -t ARGS < <(python scripts/train_act_tools.py args --steps "$STEPS" --workers "$WORKERS" --out "$OUT" \
                        --save-freq "$SAVE_FREQ" --log-freq "$LOG_FREQ")
  python -m lerobot.scripts.train "${ARGS[@]}" 2>&1 | grep --line-buffered -v -E "UserWarning|warnings.warn" | tee -a "$LOG"
fi
RC=${PIPESTATUS[0]}

python scripts/train_act_tools.py parse "$LOG" --out-dir "$LOG_DIR" --log-freq "$LOG_FREQ" || true
if [ "$RC" -eq 0 ]; then
  echo "[train_act] 训练结束。最终权重：$OUT/checkpoints/last/pretrained_model"
else
  echo "[train_act] 训练进程退出码 $RC。重跑同一命令即可从最新检查点续训。" >&2
fi
exit "$RC"
