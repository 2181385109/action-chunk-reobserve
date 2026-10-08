"""T10b 自训练 ACT 正式实验网格（PREREGISTRATION_OWN.md 的实现）：基础水平测试 + 3 种执行长度 k × 3 种条件 = 9 格。
纯计算，不依赖仿真，单元测试覆盖。

格子名与 ACT 正式网格相同（"<策略>__<条件>"，如 k20__push0.06、k100__delay10、k50__none），
所以同一份分析代码既能读 results/main_own/（自训练模型），也能读 results/main/ 里预训练模型的对应 9 格（演示运行）。
检查点路径和哈希写死为登记的第 100000 步（登记文件 §2）；run_main_own.sh 开跑前核对哈希。
"""
from __future__ import annotations

from pathlib import Path

from .pilot import Cell, config_yaml as _pilot_config_yaml

# 登记的检查点（PREREGISTRATION_OWN.md §2）：最终检查点，看到任何评估结果之前就定下，不挑选
CHECKPOINT = "/root/train_out/act_own/checkpoints/100000/pretrained_model"
CHECKPOINT_SHA256 = "28631d3abc89d054d5d26460bf604377db3c3f054138c4e3c588cd6bd6bda4d1"   # model.safetensors

OWN_SEEDS = "200000-200199"   # 与预训练模型实验同一批种子；每格 200 回合，所有格子同一批（按种子配对）
OWN_N = 200
OWN_BATCH = 10
BASELINE_SEEDS = "1000-1499"  # 基础水平测试：与预训练模型复现（results/repro/act）同一协议
BASELINE_N = 500
BASELINE_K = 100
BASELINE_NAME = "own_baseline"
OFFICIAL_SUCCESS = 0.830      # 官方报告（docs/ENVIRONMENT.md）
WEAK_THRESHOLD = 0.40         # 基础水平低于它：README 注明“自训练模型较弱”（登记 §3）

POLICIES = [("k20", "fixed", 20), ("k50", "fixed", 50), ("k100", "fixed", 100)]
PERTURBS = [
    ("none", "none", [0]),
    ("push", "push", [0.06]),          # 米；窗口 [20, 100]（与 ACT 正式实验相同，pilot.PILOT_PUSH_PARAMS）
    ("delay", "action_delay", [10]),   # 步（50 Hz：200 ms）
]
# 两个主要假设（A 的成功率高于 B；双侧检验；Holm 族 = 这两个）：(名字, 条件标签, 策略 A, 策略 B, 对应的 ACT 假设)
HYPOTHESES = [
    ("H1'", "push0.06", "k20", "k100", "H1"),
    ("H3'", "delay10", "k100", "k50", "H3"),
]
# 探索性差距变化（登记 §6.1）：短 k、长 k、扰动条件
GAP = dict(short="k20", long="k100", conds=["push0.06", "delay10"])


def condition_tags() -> list[str]:
    return ["none" if kind == "none" else f"{short}{s:g}" for short, kind, levels in PERTURBS for s in levels]


def cells() -> list[Cell]:
    """9 格，按“条件 → 策略”排序（先把一个条件的所有 k 跑完，中途停下也有完整的一块）。"""
    out = []
    for short, kind, levels in PERTURBS:
        for s in levels:
            for pol, mode, k in POLICIES:
                tag = "none" if kind == "none" else f"{short}{s:g}"
                out.append(Cell(f"{pol}__{tag}", pol, mode, k, short, kind, float(s)))
    return out


def _with_checkpoint(text: str) -> str:
    return text.rstrip("\n") + f"\npolicy_path: {CHECKPOINT}\n"


def config_yaml(c: Cell, seeds: str = OWN_SEEDS, batch_size: int = OWN_BATCH) -> str:
    text = _pilot_config_yaml(c, seeds, batch_size)
    text = text.replace("# T6a 试跑格子", "# T10b 自训练 ACT 正式评估格子", 1).replace(
        "scripts/pilot_tools.py gen", "scripts/main_own_tools.py gen", 1)
    return _with_checkpoint(text)


def baseline_yaml() -> str:
    lines = [f"# T10b 自训练 ACT 基础水平测试（由 scripts/main_own_tools.py gen 生成，不要手改）",
             f"name: {BASELINE_NAME}", "model: act", f"seeds: {BASELINE_SEEDS}", f"batch_size: {OWN_BATCH}",
             "mode: fixed", f"k: {BASELINE_K}"]
    return _with_checkpoint("\n".join(lines) + "\n")


def write_configs(out_dir: Path, seeds: str = OWN_SEEDS, batch_size: int = OWN_BATCH) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in cells():
        p = out_dir / f"{c.name}.yaml"
        p.write_text(config_yaml(c, seeds, batch_size), encoding="utf-8", newline="\n")
        paths.append(p)
    p = out_dir / f"{BASELINE_NAME}.yaml"
    p.write_text(baseline_yaml(), encoding="utf-8", newline="\n")
    paths.append(p)
    return paths


def parse_selection(spec: str | None, names: list[str]) -> list[str]:
    """子集选择：None / "" / "all" = 全部；"1-5" = 第 1 到第 5 格（从 1 数，含两端）；"a,b,c" = 按格子名。"""
    if not spec or spec == "all":
        return list(names)
    if spec.replace("-", "").isdigit() and spec.count("-") <= 1:
        lo_s, _, hi_s = spec.partition("-")
        lo, hi = int(lo_s), int(hi_s or lo_s)
        if not (1 <= lo <= hi <= len(names)):
            raise ValueError(f"格子范围 {spec} 不合法（共 {len(names)} 格，从 1 数）")
        return names[lo - 1:hi]
    want = spec.split(",")
    bad = [w for w in want if w not in names]
    if bad:
        raise ValueError(f"未知格子名：{bad}")
    return want


__all__ = ["CHECKPOINT", "CHECKPOINT_SHA256", "OWN_SEEDS", "OWN_N", "OWN_BATCH", "BASELINE_SEEDS", "BASELINE_N",
           "BASELINE_K", "BASELINE_NAME", "OFFICIAL_SUCCESS", "WEAK_THRESHOLD", "POLICIES", "PERTURBS", "HYPOTHESES",
           "GAP", "Cell", "condition_tags", "cells", "config_yaml", "baseline_yaml", "write_configs", "parse_selection"]
