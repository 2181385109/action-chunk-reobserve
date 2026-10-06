#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T4: bash scripts/t4_py.sh window | analyze   (scripts/t4_checks.py 的子命令)
source scripts/env.sh
python scripts/t4_checks.py "$@"
