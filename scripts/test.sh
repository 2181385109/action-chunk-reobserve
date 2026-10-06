#!/bin/bash
# 单元测试（不跑仿真）：bash scripts/test.sh
source /mnt/d/xiangmu/chunk-reobserve/scripts/env.sh
python -m pytest -q "$@"
