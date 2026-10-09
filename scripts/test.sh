#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# 单元测试（不跑仿真）：bash scripts/test.sh
source scripts/env.sh
python -m pytest -q "$@" || exit $?
python scripts/check_readme_numbers.py   # README 数字与 results/ 一致性检查（T12），输出 results/readme_check.txt
