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

# --- 渲染后端（实测结论见 docs/ENVIRONMENT.md）---
# WSL 里 MUJOCO_GL=egl 默认落到 Mesa llvmpipe（CPU 软渲染，aloha ~67 ms/步）；osmesa 同样是 CPU（~66 ms/步）。
# 用 Mesa 的 D3D12 驱动可以走 NVIDIA GPU 渲染（aloha ~12 ms/步）。默认用 GPU；
# 想退回 CPU 软渲染：export CHUNK_RENDER=cpu 再 source。注意：不同渲染器像素有细微差别，
# 同一个研究里所有正式结果必须用同一种（默认 gpu）。
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
export CHUNK_RENDER="${CHUNK_RENDER:-gpu}"
if [ "$CHUNK_RENDER" = "gpu" ]; then
  export GALLIUM_DRIVER=d3d12
  export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
else
  unset GALLIUM_DRIVER MESA_D3D12_DEFAULT_ADAPTER_NAME
fi

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
