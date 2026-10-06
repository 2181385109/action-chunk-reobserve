#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# 对比 egl / osmesa 渲染耗时，并测 DP 一次动作块预测耗时。结果粘贴进 docs/ENVIRONMENT.md。
source /mnt/d/xiangmu/chunk-reobserve/scripts/env.sh
for gl in egl osmesa; do
  echo "=== MUJOCO_GL=$gl"
  MUJOCO_GL=$gl PYOPENGL_PLATFORM=$gl python scripts/bench_render.py aloha 100 2>&1 | grep -E "ms/step|Error|error" | tail -3
done
echo "=== DP inference"
python scripts/bench_render.py dp_infer 2>&1 | grep -E "DP:|params|Error" | tail -3
echo "=== EGL renderer"
MUJOCO_GL=egl python - <<'PY' 2>&1 | tail -3
import mujoco, os
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 64, 64)
import OpenGL.GL as gl
print("GL_RENDERER:", gl.glGetString(gl.GL_RENDERER), "GL_VERSION:", gl.glGetString(gl.GL_VERSION))
PY
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv,noheader
