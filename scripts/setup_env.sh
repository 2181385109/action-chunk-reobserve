#!/bin/bash
# 可重复运行的环境安装脚本（在 WSL 内、root 下运行）。
# 用法：bash scripts/setup_env.sh            # 使用下面固定的版本
#       LEROBOT_VERSION=0.4.4 bash scripts/setup_env.sh   # 试别的版本（会装到同一个 venv）
set -euo pipefail

CHUNK_ROOT="/mnt/d/xiangmu/chunk-reobserve"
CHUNK_VENV="/root/venvs/chunk-reobserve"
LEROBOT_VERSION="${LEROBOT_VERSION:-PLACEHOLDER}"
# extras 名称随版本不同，若 setup 失败见 docs/ENVIRONMENT.md
LEROBOT_EXTRAS="${LEROBOT_EXTRAS:-aloha,pusht}"

export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_DEFAULT_INDEX="$PIP_INDEX_URL"
export UV_INDEX_URL="$PIP_INDEX_URL"

echo "[setup] 系统依赖（EGL/OSMesa/ffmpeg 等，已装则跳过）"
need=""
for p in libegl1 libgl1 libgles2 libosmesa6 libglfw3 ffmpeg git build-essential cmake pkg-config libglib2.0-0 python3.12-venv python3-pip; do
  dpkg -s "$p" >/dev/null 2>&1 || need="$need $p"
done
if [ -n "$need" ]; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $need
fi

mkdir -p /root/venvs
if [ ! -x "$CHUNK_VENV/bin/python" ]; then
  echo "[setup] 创建 venv（系统 Python 3.12）"
  python3 -m venv "$CHUNK_VENV"
fi
# shellcheck disable=SC1091
source "$CHUNK_VENV/bin/activate"
python -m pip install -q -U pip uv
export UV_LINK_MODE=copy   # /root 与 /mnt/d 不在同一文件系统；缓存在 /root 内，其实可 hardlink，保险起见 copy

echo "[setup] 安装 lerobot==$LEROBOT_VERSION[$LEROBOT_EXTRAS]"
# 先装 CUDA 版 PyTorch（驱动 592.27 支持 CUDA 12.x/13.x；用 cu124 或 cu128 轮子均可，由 TORCH_INDEX 控制）
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
if [ -n "${TORCH_SPEC:-}" ]; then
  uv pip install $TORCH_SPEC --index-url "$TORCH_INDEX" --extra-index-url "$PIP_INDEX_URL"
fi
uv pip install "lerobot[$LEROBOT_EXTRAS]==$LEROBOT_VERSION"

echo "[setup] 完成。版本清单："
python - <<'PY'
import importlib.metadata as m
for p in ["lerobot","torch","torchvision","numpy","gymnasium","gym-aloha","gym-pusht","mujoco","dm-control","huggingface-hub","transformers","diffusers","safetensors"]:
    try: print(f"  {p}=={m.version(p)}")
    except Exception as e: print(f"  {p}: 未安装")
import torch
print("  torch.cuda.is_available() =", torch.cuda.is_available(), "| torch.version.cuda =", torch.version.cuda)
PY
