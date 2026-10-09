"""T8b DP 试跑网格：4 种执行长度 k ∈ {2, 4, 8, 15}（fixed）× 10 种扰动条件 = 40 格（PushT，lerobot/diffusion_pusht）。
纯计算，不依赖仿真，单元测试覆盖。

只在调参种子上描述 DP 的 k × 扰动关系，不作为结论。格子名 = "<策略>__<扰动>"，如 k8__none、k2__push40、k15__noise16、k4__delay2。
推 T 形块的档位 20 / 40 / 60（PushT 坐标，T 形块整体 120×120）和窗口 [20, 80] 的依据见 results/t8_dp_baseline/baseline.md；
噪声 σ 与 ACT 正式实验相同的绝对档位 4 / 8 / 16（0–255）；延迟 1 / 2 / 4 步（10 Hz：100 / 200 / 400 ms，T5 建议）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .pilot import FLOOR, CEIL, cell_done, flag_of  # noqa: F401  （与 ACT 试跑同一套完成判断和地板 / 天花板阈值）

PILOT_DP_SEEDS = "5000-5049"   # 调参区间（0–9999）；每格 50 回合，有意固定，不要改
PILOT_DP_BATCH = 50            # DP 用 B=50（T2 耗时表：B 越大每回合越快）
PUSH_PARAMS = dict(t_lo=20, t_hi=80)   # 与 chunkre.perturb.PushTee.DEFAULTS、configs/dp_push.yaml 一致

# (标签, mode, k)
POLICIES = [("k2", "fixed", 2), ("k4", "fixed", 4), ("k8", "fixed", 8), ("k15", "fixed", 15)]
# (标签前缀, perturb.kind, 档位)；none 只有一个条件
PERTURBS = [
    ("none", "none", [0]),
    ("push", "push", [20, 40, 60]),
    ("noise", "obs_noise", [4, 8, 16]),
    ("delay", "action_delay", [1, 2, 4]),
]
CSV_COLUMNS = ["cell", "policy", "perturb", "strength", "n", "n_success", "success_rate", "ci_lo", "ci_hi",
               "mean_max_reward", "mean_policy_calls", "mean_path_length", "mean_jerk", "mean_steps", "flag",
               "n_pushed", "n_skipped_overlap", "n_not_reached", "n_clip_wall"]


@dataclass(frozen=True)
class Cell:
    name: str
    policy: str          # k2 / k4 / k8 / k15
    mode: str            # fixed
    k: int
    perturb: str         # none / push / noise / delay
    kind: str            # perturb.kind
    strength: float      # 推：PushT 坐标；噪声：σ；延迟：步数；none：0

    @property
    def perturb_tag(self) -> str:
        return "none" if self.kind == "none" else f"{self.perturb}{self.strength:g}"


def cells() -> list[Cell]:
    """40 格，按“扰动类型 → 强度 → 策略”排序（中途停下也有完整的一块）。"""
    out = []
    for short, kind, levels in PERTURBS:
        for s in levels:
            for pol, mode, k in POLICIES:
                tag = "none" if kind == "none" else f"{short}{s:g}"
                out.append(Cell(f"{pol}__{tag}", pol, mode, k, short, kind, float(s)))
    return out


def config_yaml(c: Cell, seeds: str = PILOT_DP_SEEDS, batch_size: int = PILOT_DP_BATCH) -> str:
    lines = [
        f"# T8b DP 试跑格子 {c.name}（由 scripts/pilot_tools.py --grid dp gen 生成，不要手改）",
        f"name: {c.name}",
        "model: dp",
        f"seeds: {seeds}",
        f"batch_size: {batch_size}",
        f"mode: {c.mode}",
        f"k: {c.k}",
    ]
    if c.kind != "none":
        lines += ["perturb:", f"  kind: {c.kind}", f"  strength: {c.strength:g}"]
        if c.kind == "push":
            lines += ["  params:"] + [f"    {k}: {v}" for k, v in PUSH_PARAMS.items()]
    return "\n".join(lines) + "\n"


def write_configs(out_dir: Path, seeds: str = PILOT_DP_SEEDS, batch_size: int = PILOT_DP_BATCH) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in cells():
        p = out_dir / f"{c.name}.yaml"
        p.write_text(config_yaml(c, seeds, batch_size), encoding="utf-8", newline="\n")
        paths.append(p)
    return paths
