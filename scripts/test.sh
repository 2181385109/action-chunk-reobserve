#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# 单元测试（不跑仿真）：bash scripts/test.sh
source /mnt/d/xiangmu/chunk-reobserve/scripts/env.sh
python -m pytest -q "$@"
