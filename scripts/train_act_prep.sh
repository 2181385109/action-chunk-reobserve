#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T10a：bash scripts/train_act_prep.sh download | bench --workers 4 --steps 600 | diffcfg <train_config.json>
source scripts/env.sh
python scripts/train_act_prep.py "$@" 2>&1 | grep -v -E "UserWarning|logger.warn|warnings.warn"
exit "${PIPESTATUS[0]}"
