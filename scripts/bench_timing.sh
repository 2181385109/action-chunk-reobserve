#!/bin/bash
# V5 耗时基准（约 3–4 分钟）。必须单独跑（不要与其他评估并行，否则耗时不准）。
# 用法：bash scripts/bench_timing.sh   [BS=1,10,25,50]
source /mnt/d/xiangmu/chunk-reobserve/scripts/env.sh
OUT=results/t2_checks
mkdir -p $OUT
rm -f $OUT/timing_raw_act.csv $OUT/timing_raw_dp.csv
BS="${BS:-1,10,25,50}"
# 一个 B 一个 B 地跑：某个 B 内存不够被杀时，较小 B 的结果已写入，并停在能跑的最大值
for m in act dp; do
  for b in ${BS//,/ }; do
    python scripts/bench_timing.py measure $m --Bs $b 2>&1 | grep -E "\[bench|Error|Killed|error" \
      || { echo "[bench $m] B=$b 失败（可能内存不够），停在更小的 B"; break; }
    if ! grep -q "^$m,$b," $OUT/timing_raw_$m.csv 2>/dev/null; then
      echo "[bench $m] B=$b 没有写出结果（可能被内存不足杀掉），停在更小的 B"; break
    fi
  done
done
# 对照：ACT 在 CPU 软渲染（llvmpipe，逐字节可复现）下的耗时，供统筹决定渲染器时参考
rm -f $OUT/timing_raw_act_cpu.csv
export CHUNK_RENDER=cpu
source /mnt/d/xiangmu/chunk-reobserve/scripts/env.sh
for b in ${BS_CPU:-1,10}; do
  for bb in ${b//,/ }; do
    python scripts/bench_timing.py measure act --variant _cpu --Bs $bb 2>&1 | grep -E "\[bench|Error|Killed" || break
  done
done
python scripts/bench_timing.py estimate
