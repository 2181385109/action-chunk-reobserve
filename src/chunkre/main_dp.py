"""T9 DP 正式实验网格（PREREGISTRATION_DP.md 的实现）：3 种执行长度 k × 4 种条件 = 12 格（PushT，lerobot/diffusion_pusht）。
纯计算，不依赖仿真，单元测试覆盖。

格子名与 DP 试跑相同（"<策略>__<条件>"，如 k4__push60、k15__delay4、k8__none），所以同一份分析代码既能读
results/main_dp/（正式，种子 200000–200199），也能读 results/pilot_dp/（演示运行，调参种子 5000–5049）。
噪声不进正式网格（登记文件 §3 写明原因）。
"""
from __future__ import annotations

from pathlib import Path

from .pilot_dp import PUSH_PARAMS, Cell, config_yaml as _pilot_config_yaml

MAIN_DP_SEEDS = "200000-200199"   # 事先登记：每格 200 回合，所有格子同一批种子（按种子配对）
MAIN_DP_N = 200
MAIN_DP_BATCH = 50                 # B=50（每格 4 批）
POLICIES = [("k4", "fixed", 4), ("k8", "fixed", 8), ("k15", "fixed", 15)]
PERTURBS = [
    ("none", "none", [0]),
    ("push", "push", [60]),          # PushT 坐标；窗口 [20, 80]，重叠时重抽方向（PUSH_PARAMS / PushTee.DEFAULTS）
    ("delay", "action_delay", [2, 4]),   # 步（10 Hz：200 / 400 ms）
]
# 两个主要假设（A 的成功率高于 B；双侧检验；Holm 族 = 这两个）：(名字, 条件标签, 策略 A, 策略 B, 对应的 ACT 假设)
HYPOTHESES = [
    ("H4", "push60", "k4", "k15", "H1"),
    ("H5", "delay4", "k15", "k8", "H3"),
]


def condition_tags() -> list[str]:
    return ["none" if kind == "none" else f"{short}{s:g}" for short, kind, levels in PERTURBS for s in levels]


def cells() -> list[Cell]:
    """12 格，按“条件 → 策略”排序（先把一个条件的所有 k 跑完，中途停下也有完整的一块）。"""
    out = []
    for short, kind, levels in PERTURBS:
        for s in levels:
            for pol, mode, k in POLICIES:
                tag = "none" if kind == "none" else f"{short}{s:g}"
                out.append(Cell(f"{pol}__{tag}", pol, mode, k, short, kind, float(s)))
    return out


def config_yaml(c: Cell, seeds: str = MAIN_DP_SEEDS, batch_size: int = MAIN_DP_BATCH) -> str:
    text = _pilot_config_yaml(c, seeds, batch_size)
    return text.replace("# T8b DP 试跑格子", "# T9 DP 正式评估格子", 1).replace(
        "scripts/pilot_tools.py --grid dp gen", "scripts/main_dp_tools.py gen", 1)


def write_configs(out_dir: Path, seeds: str = MAIN_DP_SEEDS, batch_size: int = MAIN_DP_BATCH) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in cells():
        p = out_dir / f"{c.name}.yaml"
        p.write_text(config_yaml(c, seeds, batch_size), encoding="utf-8", newline="\n")
        paths.append(p)
    return paths


def parse_selection(spec: str | None, names: list[str]) -> list[str]:
    """子集选择：None / "" / "all" = 全部；"1-6" = 第 1 到第 6 格（从 1 数，含两端）；"a,b,c" = 按格子名。"""
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


__all__ = ["MAIN_DP_SEEDS", "MAIN_DP_N", "MAIN_DP_BATCH", "POLICIES", "PERTURBS", "HYPOTHESES", "PUSH_PARAMS",
           "Cell", "condition_tags", "cells", "config_yaml", "write_configs", "parse_selection"]
