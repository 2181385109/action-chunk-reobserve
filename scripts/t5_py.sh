#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T5: bash scripts/t5_py.sh n2 [--models act,dp] | analyze   (scripts/t5_checks.py 的子命令)
source scripts/env.sh
python scripts/t5_checks.py "$@" 2>&1 | grep -v -E "UserWarning|logger.warn|task_description|deprecated|warnings.warn"
