"""T6a 试跑网格：5 种策略 × 11 种扰动条件 = 55 格，加 T6b 补测 σ=32（k=20/50/100/te，k=10 不测）4 格 = 59 格（只 aloha ACT）。纯计算，不依赖仿真，单元测试覆盖。

只用于在调参种子上粗跑、定正式实验的强度档位和每格回合数，不作为结论。
格子名 = "<策略>__<扰动>"，如 k10__push0.02、te__noise8、k100__delay5、k50__none。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

PILOT_SEEDS = "5000-5019"   # 调参区间（0–9999）内此前没用过的种子，20 回合；有意固定，不要改
PILOT_BATCH = 10
PILOT_PUSH_PARAMS = dict(t_lo=20, t_hi=100)   # 与 configs/act_push.yaml 一致（T4 定的推的窗口）

# (标签, mode, k)
POLICIES = [("k10", "fixed", 10), ("k20", "fixed", 20), ("k50", "fixed", 50), ("k100", "fixed", 100), ("te", "te", None)]
# (标签前缀, perturb.kind, 档位)；none 只有一个条件
PERTURBS = [
    ("none", "none", [0]),
    ("push", "push", [0.02, 0.04, 0.06]),
    ("noise", "obs_noise", [4, 8, 16, 32]),   # σ=32 是 T6b 补测（只测 k20/k50/k100/te，见 SKIP）
    ("delay", "action_delay", [1, 3, 5, 10]),
]
# T6b 补测 σ=32 不测 k=10（k=10 在所有条件下都在地板，没有信息）
SKIP = {("k10", "noise32")}
CSV_COLUMNS = ["cell", "policy", "perturb", "strength", "n", "n_success", "success_rate", "ci_lo", "ci_hi",
               "mean_policy_calls", "mean_path_length", "mean_jerk", "mean_steps", "flag"]
FLOOR, CEIL = 0.10, 0.90   # 成功率 <= FLOOR 记“地板”，>= CEIL 记“天花板”


@dataclass(frozen=True)
class Cell:
    name: str
    policy: str          # k10 / k20 / k50 / k100 / te
    mode: str            # fixed / te
    k: int | None        # te 时为 None
    perturb: str         # none / push / noise / delay（图和表用的短名）
    kind: str            # perturb.kind：none / push / obs_noise / action_delay
    strength: float      # 推方块：米；噪声：σ；延迟：步数；none：0

    @property
    def perturb_tag(self) -> str:
        return "none" if self.kind == "none" else f"{self.perturb}{self.strength:g}"


def cells() -> list[Cell]:
    """59 格，按“扰动类型 → 强度 → 策略”排序（先把一种扰动的所有策略跑完，中途停下也有完整的一块）。"""
    out = []
    for short, kind, levels in PERTURBS:
        for s in levels:
            for pol, mode, k in POLICIES:
                tag = "none" if kind == "none" else f"{short}{s:g}"
                if (pol, tag) in SKIP:
                    continue
                out.append(Cell(f"{pol}__{tag}", pol, mode, k, short, kind, float(s)))
    return out


def config_yaml(c: Cell, seeds: str = PILOT_SEEDS, batch_size: int = PILOT_BATCH) -> str:
    lines = [
        f"# T6a 试跑格子 {c.name}（由 scripts/pilot_tools.py gen 生成，不要手改）",
        f"name: {c.name}",
        "model: act",
        f"seeds: {seeds}",
        f"batch_size: {batch_size}",
        f"mode: {c.mode}",
    ]
    if c.k is not None:
        lines.append(f"k: {c.k}")
    if c.kind != "none":
        lines += ["perturb:", f"  kind: {c.kind}", f"  strength: {c.strength:g}"]
        if c.kind == "push":
            lines += ["  params:"] + [f"    {k}: {v}" for k, v in PILOT_PUSH_PARAMS.items()]
    return "\n".join(lines) + "\n"


def write_configs(out_dir: Path, seeds: str = PILOT_SEEDS, batch_size: int = PILOT_BATCH) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in cells():
        p = out_dir / f"{c.name}.yaml"
        p.write_text(config_yaml(c, seeds, batch_size), encoding="utf-8", newline="\n")
        paths.append(p)
    return paths


def cell_done(cell_dir: Path, n_expected: int) -> bool:
    """格子已完成 = summary.json 在，且 episodes.csv 恰好有 n_expected 个回合（run_eval 只在跑完后才写 summary.json）。"""
    cell_dir = Path(cell_dir)
    csv_path = cell_dir / "episodes.csv"
    if not (cell_dir / "summary.json").exists() or not csv_path.exists():
        return False
    with open(csv_path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f)) == n_expected


def flag_of(rate: float) -> str:
    if rate != rate:   # nan
        return ""
    if rate <= FLOOR:
        return "floor"
    if rate >= CEIL:
        return "ceiling"
    return ""
