"""T6b 正式实验网格（PREREGISTRATION.md 的实现）：4 种策略 × 10 种条件 = 40 格（只 aloha ACT）。纯计算，不依赖仿真。

格子名与试跑相同（"<策略>__<扰动>"，如 k20__push0.06、te__noise16、k100__delay10、k50__none），
所以同一份分析代码既能读 results/main/（正式，种子 200000–200199），也能读 results/pilot/（演示运行，调参种子）。
"""
from __future__ import annotations

from pathlib import Path

from .pilot import Cell, config_yaml as _pilot_config_yaml

MAIN_SEEDS = "200000-200199"   # 事先登记：每格 200 回合，所有格子同一批种子（按种子配对）
MAIN_N = 200
MAIN_BATCH = 10
PUSH_LEVELS = [0.02, 0.04, 0.06]       # 米；窗口 [20, 100]（与试跑相同，见 pilot.PILOT_PUSH_PARAMS）
NOISE_LEVELS = [4, 8, 16]               # σ（0–255）：按 T6b 第 1 步的机械规则选出
DELAY_LEVELS = [3, 5, 10]              # 步（50 Hz：60 / 100 / 200 ms）
POLICIES = [("k20", "fixed", 20), ("k50", "fixed", 50), ("k100", "fixed", 100), ("te", "te", None)]
PERTURBS = [
    ("none", "none", [0]),
    ("push", "push", PUSH_LEVELS),
    ("noise", "obs_noise", NOISE_LEVELS),
    ("delay", "action_delay", DELAY_LEVELS),
]
# 三个主要假设（A 的成功率高于 B；双侧检验）：(名字, 条件标签, 策略 A, 策略 B)。条件标签与格子名后缀一致。
STRONGEST_NOISE = max(NOISE_LEVELS)
HYPOTHESES = [
    ("H1", f"push{PUSH_LEVELS[-1]:g}", "k20", "k100"),
    ("H2", f"noise{STRONGEST_NOISE:g}", "k100", "te"),
    ("H3", f"delay{DELAY_LEVELS[-1]:g}", "k100", "k50"),
]


def condition_tags() -> list[str]:
    """10 个条件标签，按“扰动类型 → 强度”排序：none, push0.02, …, noise.., delay.."""
    return ["none" if kind == "none" else f"{short}{s:g}" for short, kind, levels in PERTURBS for s in levels]


def cells() -> list[Cell]:
    """40 格，按“扰动类型 → 强度 → 策略”排序（与试跑相同；中途停下总有完整的条件）。"""
    out = []
    for short, kind, levels in PERTURBS:
        for s in levels:
            for pol, mode, k in POLICIES:
                tag = "none" if kind == "none" else f"{short}{s:g}"
                out.append(Cell(f"{pol}__{tag}", pol, mode, k, short, kind, float(s)))
    return out


def config_yaml(c: Cell, seeds: str = MAIN_SEEDS, batch_size: int = MAIN_BATCH) -> str:
    text = _pilot_config_yaml(c, seeds, batch_size)
    return text.replace("# T6a 试跑格子", "# T6b 正式评估格子", 1)


def write_configs(out_dir: Path, seeds: str = MAIN_SEEDS, batch_size: int = MAIN_BATCH) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in cells():
        p = out_dir / f"{c.name}.yaml"
        p.write_text(config_yaml(c, seeds, batch_size), encoding="utf-8", newline="\n")
        paths.append(p)
    return paths


def parse_selection(spec: str | None, names: list[str]) -> list[str]:
    """子集选择：None / "" / "all" = 全部；"1-20" = 第 1 到第 20 格（从 1 数，含两端）；"a,b,c" = 按格子名。"""
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
