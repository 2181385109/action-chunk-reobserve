#!/bin/bash
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." || exit 1   # run from repo root, whatever the caller cwd
# T11b 参数核对拦截演示（不跑回合）：用环境变量把 tau 改成 0.3（登记是 0.2906），run_main_adaptive.sh 应在加载模型之前退出（exit 4）；
# 配置写到临时目录，输出目录也指到临时目录（不动 configs/main_adaptive 和 results/main_adaptive）。输出见 results/params_guard_demo_adaptive.txt
TMP=$(mktemp -d)
echo "# 1) tau 改成 0.3（登记 0.2906）"
MAIN_ADAPTIVE_CONFIG_DIR="$TMP/cfg" MAIN_ADAPTIVE_OUT="$TMP/out" ADAPT_TAU=0.3 bash scripts/run_main_adaptive.sh 2>&1
echo "退出码 $?（期望 4）；输出目录是否被创建：$([ -d "$TMP/out" ] && echo 是 || echo 否)"
echo
echo "# 2) m 改成 20（登记 10）"
MAIN_ADAPTIVE_CONFIG_DIR="$TMP/cfg" MAIN_ADAPTIVE_OUT="$TMP/out" ADAPT_M=20 bash scripts/run_main_adaptive.sh 2>&1
echo "退出码 $?（期望 4）；输出目录是否被创建：$([ -d "$TMP/out" ] && echo 是 || echo 否)"
echo
echo "# 3) min_gap 改成 10（登记 20）"
MAIN_ADAPTIVE_CONFIG_DIR="$TMP/cfg" MAIN_ADAPTIVE_OUT="$TMP/out" ADAPT_MIN_GAP=10 bash scripts/run_main_adaptive.sh 2>&1
echo "退出码 $?（期望 4）；输出目录是否被创建：$([ -d "$TMP/out" ] && echo 是 || echo 否)"
rm -rf "$TMP"
