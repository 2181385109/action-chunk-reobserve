#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T10a：在 WSL 环境里跑任意 python 命令，如 bash scripts/t10a_py.sh scripts/summarize.py <目录>
source scripts/env.sh
python "$@"
