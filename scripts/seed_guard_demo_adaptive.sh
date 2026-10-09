#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T11b seed_guard 拦截演示（不跑回合）。输出见 results/seed_guard_demo_adaptive.txt
source scripts/env.sh
python scripts/seed_guard_demo_adaptive.py
