#!/bin/bash
# T2 ACT 补充检查（发现 D3D12 GPU 渲染不逐字节可复现之后）：
#  1) GPU 渲染（研究默认）：V4（k=1、10、te；k=100 用 act_v1_run1）——调用次数检查不依赖逐位可复现；
#  2) CPU 渲染 llvmpipe（逐字节可复现）：V1 两次、V2、V3 四个进程并行，用来检验评估脚本本身。
# 用法：bash scripts/t2_checks_act.sh   （约 8 分钟）
source /mnt/d/xiangmu/chunk-reobserve/scripts/env.sh
OUT=results/t2_checks
rm -rf $OUT/act_v4_* $OUT/act_cpu_*
python scripts/t2_checks.py run act --only v4 > $OUT/act_v4_chain.log 2>&1 &
PG=$!

export CHUNK_RENDER=cpu
source /mnt/d/xiangmu/chunk-reobserve/scripts/env.sh
export LP_NUM_THREADS=4   # 4 个进程并行，每个 llvmpipe 用 4 线程（16 核）
python scripts/eval.py --config configs/act_default.yaml --out $OUT/act_cpu_v1_run1 --seeds 0-9 --batch-size 10 \
  --allow-cpu-render > $OUT/act_cpu_v1a.log 2>&1 &
P1=$!
python scripts/t2_checks.py run act --cpu --only v1b > $OUT/act_cpu_v1b.log 2>&1 &
P2=$!
python scripts/t2_checks.py run act --cpu --only v2 > $OUT/act_cpu_v2.log 2>&1 &
P3=$!
python scripts/t2_checks.py run act --cpu --only v3 > $OUT/act_cpu_v3.log 2>&1 &
P4=$!
fail=0
for p in $PG $P1 $P2 $P3 $P4; do wait $p || fail=1; done
grep -ah "批 \|V3\|Error" $OUT/act_v4_chain.log $OUT/act_cpu_*.log
python scripts/t2_checks.py analyze > /dev/null
cat $OUT/checks.md
exit $fail
