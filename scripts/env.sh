#!/bin/bash
# 统一入口：source scripts/env.sh   （在 WSL 内使用）
# 设置：虚拟环境、HF 缓存/端点、pip 镜像、MuJoCo 离屏渲染后端、项目路径。

export CHUNK_ROOT="/mnt/d/xiangmu/chunk-reobserve"
export CHUNK_VENV="/root/venvs/chunk-reobserve"

# --- 网络（机器在中国大陆；2026-10-06 实测）---
# huggingface.co 直连 ~1s、下载 ~6MB/s，与 hf-mirror.com 相当，所以默认直连。
# 如果直连变慢/不通，运行前设置：export HF_ENDPOINT=https://hf-mirror.com
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="/root/.cache/huggingface"
# pypi.org 实测 15s 超时 → 使用清华镜像（uv 与 pip 都读这两个变量）
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export UV_INDEX_URL="$PIP_INDEX_URL"
export UV_DEFAULT_INDEX="$PIP_INDEX_URL"

# --- 渲染后端（最终选择见 docs/ENVIRONMENT.md）---
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"

# --- 其他 ---
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false

if [ -f "$CHUNK_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$CHUNK_VENV/bin/activate"
else
  echo "[env.sh] 警告：虚拟环境 $CHUNK_VENV 不存在，请先运行 scripts/setup_env.sh" >&2
fi
cd "$CHUNK_ROOT" || true
