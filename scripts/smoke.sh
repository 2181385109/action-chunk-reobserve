#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# 冒烟测试包装：bash scripts/smoke.sh [额外参数传给 smoke.py]
set -e
source /mnt/d/xiangmu/chunk-reobserve/scripts/env.sh
python scripts/smoke.py "$@" 2>&1 | tee results/smoke/smoke.log
