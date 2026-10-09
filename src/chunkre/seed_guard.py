"""种子硬性检查（CLAUDE.md §5）。所有评估/调试脚本都应在使用种子前调用 check_seeds()。

- 0–9999：调试和调参（含复现），随时可用。
- 10000–199999：保留，不使用（官方 DP 训练种子 100000，训练中的评估用过 100000 开头的种子）。
- >=200000：只用于最终评估；对应的登记文件在仓库里存在且已 git 提交之前，直接抛错。
  ACT（model=None / "act"）看 PREREGISTRATION.md；DP（model="dp"，T9）看 PREREGISTRATION_DP.md；
  自训练 ACT（model="act_own"，T10b；run_eval 在 --policy-path 不为空时用 guard_model() 得到它）看 PREREGISTRATION_OWN.md；
  自适应规则（mode="adaptive"，T11b；guard_model(model, policy_path, mode) 得到 "act_adaptive"）看 PREREGISTRATION_ADAPTIVE.md。
  自适应 + 本地检查点（登记没有覆盖）= "act_adaptive_own"，没有登记文件：>=200000 的种子直接抛错。
- 负数：不允许。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEBUG_MAX = 9999
FINAL_MIN = 200000
# 模型 -> 登记文件。model=None（不知道模型的老调用）按 ACT 处理，行为与 T2 起完全一致。
PREREG_FILES = {"act": "PREREGISTRATION.md", "dp": "PREREGISTRATION_DP.md", "act_own": "PREREGISTRATION_OWN.md",
                "act_adaptive": "PREREGISTRATION_ADAPTIVE.md"}


def guard_model(model: str | None, policy_path=None, mode: str | None = None) -> str | None:
    """评估配置 -> 种子保护用的“模型键”：mode="adaptive"（T11b）= "act_adaptive"，看 PREREGISTRATION_ADAPTIVE.md
    （自适应 + 本地检查点 = "act_adaptive_own"，没有登记，>=200000 的种子被拦）；
    否则 ACT 加载本地检查点（policy_path 不为空）= 自训练模型 "act_own"，看 PREREGISTRATION_OWN.md；
    其余照旧（hub 模型 / DP 不变）。"""
    local = policy_path not in (None, "")
    if mode == "adaptive" and model in (None, "act"):
        return "act_adaptive_own" if local else "act_adaptive"
    if local and model in (None, "act"):
        return "act_own"
    return model


def prereg_file(model: str | None) -> str:
    if model is None:
        return PREREG_FILES["act"]
    if model not in PREREG_FILES:
        raise ValueError(f"未知模型 {model!r}，没有对应的登记文件")
    return PREREG_FILES[model]


def preregistration_committed(root: Path = ROOT, model: str | None = None) -> bool:
    name = prereg_file(model)
    p = root / name
    if not p.exists():
        return False
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "log", "--oneline", "-1", "--", name],
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


def check_seeds(seeds, root: Path = ROOT, model: str | None = None) -> None:
    prereg = None
    for s in seeds:
        zone = seed_zone(s)
        if zone == "debug":
            continue
        if zone == "final":
            if prereg is None:
                prereg = preregistration_committed(root, model)
            if not prereg:
                raise RuntimeError(
                    f"种子 {s} >= {FINAL_MIN} 仅用于最终评估，而 {prereg_file(model)} 尚未提交，禁止使用。"
                )
            continue
        if zone == "reserved":
            raise RuntimeError(f"种子 {s} 落在保留区间 {DEBUG_MAX + 1}–{FINAL_MIN - 1}，不允许使用。")
        raise RuntimeError(f"种子 {s} 为负数，不允许使用。")
