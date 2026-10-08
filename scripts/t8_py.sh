#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T8b: bash scripts/t8_py.sh baseline | analyze   (scripts/t8_checks.py 的子命令)
source scripts/env.sh
python scripts/t8_checks.py "$@" 2>&1 | grep -v -E "UserWarning|logger.warn|task_description|deprecated|warnings.warn"
