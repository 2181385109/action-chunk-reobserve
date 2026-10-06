"""种子硬性检查（CLAUDE.md §5）。所有评估/调试脚本都应在使用种子前调用 check_seeds()。

- 0–9999：调试和调参（含复现），随时可用。
- 10000–199999：保留，不使用（官方 DP 训练种子 100000，训练中的评估用过 100000 开头的种子）。
- >=200000：只用于最终评估；PREREGISTRATION.md 在仓库里存在且已 git 提交之前，直接抛错。
- 负数：不允许。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEBUG_MAX = 9999
FINAL_MIN = 200000


def preregistration_committed(root: Path = ROOT) -> bool:
    p = root / "PREREGISTRATION.md"
    if not p.exists():
        return False
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "log", "--oneline", "-1", "--", "PREREGISTRATION.md"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return bool(out)
    except Exception:
        return False


def seed_zone(s: int) -> str:
    """返回 'debug' / 'reserved' / 'final' / 'invalid'。"""
    s = int(s)
    if 0 <= s <= DEBUG_MAX:
        return "debug"
    if DEBUG_MAX < s < FINAL_MIN:
        return "reserved"
    if s >= FINAL_MIN:
        return "final"
    return "invalid"


def check_seeds(seeds, root: Path = ROOT) -> None:
    prereg = None
    for s in seeds:
        zone = seed_zone(s)
        if zone == "debug":
            continue
        if zone == "final":
            if prereg is None:
                prereg = preregistration_committed(root)
            if not prereg:
                raise RuntimeError(
                    f"种子 {s} >= {FINAL_MIN} 仅用于最终评估，而 PREREGISTRATION.md 尚未提交，禁止使用。"
                )
            continue
        if zone == "reserved":
            raise RuntimeError(f"种子 {s} 落在保留区间 {DEBUG_MAX + 1}–{FINAL_MIN - 1}，不允许使用。")
        raise RuntimeError(f"种子 {s} 为负数，不允许使用。")
