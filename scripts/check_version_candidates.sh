#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# 版本选择依据：给每个候选版本建独立临时 venv（完整依赖，lerobot 基础包，不含仿真 extras），
# 看两个 hub 模型能否用官方 API（Policy.from_pretrained）直接加载。结果记录在 docs/ENVIRONMENT.md。
# 用法：bash scripts/check_version_candidates.sh 0.4.4 0.6.1
source scripts/env.sh
deactivate 2>/dev/null
for v in "$@"; do
  d=/root/venvs/probe_$v; rm -rf $d
  /root/venvs/chunk-reobserve/bin/uv venv -q --python /usr/bin/python3.12 $d
  /root/venvs/chunk-reobserve/bin/uv pip install -q --python $d/bin/python "lerobot[diffusion]==$v" 2>&1 | tail -3
  for m in act_aloha_sim_transfer_cube_human diffusion_pusht; do
    echo "=== lerobot $v : $m"
    $d/bin/python - <<PY 2>&1 | grep -vi "warn" | tail -5 | cut -c1-400
from lerobot.policies.factory import get_policy_class
from lerobot.configs.policies import PreTrainedConfig
cfg = PreTrainedConfig.from_pretrained("lerobot/$m")
cls = get_policy_class(cfg.type)
p = cls.from_pretrained("lerobot/$m")
print("POLICY WEIGHTS LOAD OK", type(p).__name__)
# >=0.4：归一化统计量在独立的 preprocessor/postprocessor 里；旧 hub 模型没有这些文件
from lerobot.policies.factory import make_pre_post_processors
pre, post = make_pre_post_processors(cfg, pretrained_path="lerobot/$m")
print("PROCESSORS LOAD OK")
PY
  done
  rm -rf $d
done
