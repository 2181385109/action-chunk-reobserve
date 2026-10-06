#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# 量化 D3D12 渲染不确定性：三个独立进程各渲染同一组画面，逐像素比较。约 1 分钟。
source /mnt/d/xiangmu/chunk-reobserve/scripts/env.sh
T=$(mktemp -d)
for r in a b c; do python scripts/probe_pixdiff.py dump $T/frames_$r.npy > /dev/null 2>&1; done
python scripts/probe_pixdiff.py compare $T/frames_a.npy $T/frames_b.npy $T/frames_c.npy
rm -rf $T
