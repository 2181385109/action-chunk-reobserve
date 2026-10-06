"""种子硬性检查（CLAUDE.md §5）。所有评估/调试脚本都应在使用种子前调用 check_seeds()。

- 0–9999：调试和调参（含复现），随时可用。
- >=100000：只用于最终评估；PREREGISTRATION.md 在仓库里存在且已 git 提交之前，直接抛错。
- 10000–99999：保留，不使用。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEBUG_MAX = 9999
FINAL_MIN = 100000


def preregistration_committed() -> bool:
    p = ROOT / "PREREGISTRATION.md"
    if not p.exists():
        return False
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "log", "--oneline", "-1", "--", "PREREGISTRATION.md"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return bool(out)
    except Exception:
        return False


def check_seeds(seeds) -> None:
    for s in seeds:
        s = int(s)
        if 0 <= s <= DEBUG_MAX:
            continue
        if s >= FINAL_MIN:
            if not preregistration_committed():
                raise RuntimeError(
                    f"种子 {s} >= {FINAL_MIN} 仅用于最终评估，而 PREREGISTRATION.md 尚未提交，禁止使用。"
                )
            continue
        raise RuntimeError(f"种子 {s} 落在保留区间 {DEBUG_MAX + 1}–{FINAL_MIN - 1}，不允许使用。")
