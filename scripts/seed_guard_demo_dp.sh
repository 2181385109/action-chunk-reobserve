#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T9 seed_guard 拦截演示（不跑回合）。输出见 results/seed_guard_demo_dp.txt
source scripts/env.sh
python scripts/seed_guard_demo_dp.py
