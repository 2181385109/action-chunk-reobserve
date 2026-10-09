"""兼容入口：种子检查的实现已移到 src/chunkre/seed_guard.py（T2）。

区间：0–9999 调试/调参（含复现）；10000–199999 保留不用；>=200000 只用于最终评估
（PREREGISTRATION.md 提交前禁止）。详见 src/chunkre/seed_guard.py。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from chunkre.seed_guard import (  # noqa: E402,F401
    DEBUG_MAX, FINAL_MIN, PREREG_FILES, check_seeds, preregistration_committed, prereg_file, seed_zone,
)
