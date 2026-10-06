"""纯计算：平滑度、Wilson 置信区间、从逐回合行重新算汇总。不依赖仿真，单元测试覆盖。"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

# 逐回合 CSV 的列（顺序固定）。TIME_COLUMNS 是墙钟类列，复现性比较时排除。
EPISODE_COLUMNS = [
    "batch_idx", "env_idx", "seed", "model", "mode", "k",
    "success", "steps", "max_steps", "capped",
    "sum_reward", "max_reward", "n_policy_calls", "smoothness",
    "wall_s", "batch_wall_s", "batch_policy_s", "batch_env_s",
]
TIME_COLUMNS = ["wall_s", "batch_wall_s", "batch_policy_s", "batch_env_s"]


def action_smoothness(actions) -> float:
    """动作平滑度 = 相邻两步动作差的 L2 范数的平均值。

    actions: 形状 (T, action_dim)，按执行顺序排列的、实际发给环境的动作（环境的动作单位，未归一化）。
    返回 mean_{t=1..T-1} ||a_t - a_{t-1}||_2；T < 2 时返回 nan。数值越小越平滑。
    """
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] < 2:
        return float("nan")
    return float(np.linalg.norm(np.diff(a, axis=0), axis=1).mean())


def wilson_ci(n_success: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """成功率的 Wilson 得分区间（默认 95%）。n=0 时返回 (nan, nan)。"""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = n_success / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _f(x) -> float:
    return float(x) if x not in ("", None) else float("nan")


def summarize_rows(rows: list[dict]) -> dict:
    """由逐回合行（csv.DictReader 读出的字符串或原始数值均可）计算汇总。

    summary.json 只由此函数生成，scripts/summarize.py 也调用它，所以汇总总能从 CSV 重算。
    """
    n = len(rows)
    succ = [int(_f(r["success"])) for r in rows]
    n_success = int(sum(succ))
    lo, hi = wilson_ci(n_success, n)
    calls = np.array([_f(r["n_policy_calls"]) for r in rows])
    smooth = np.array([_f(r["smoothness"]) for r in rows])
    steps = np.array([_f(r["steps"]) for r in rows])
    capped = int(sum(int(_f(r["capped"])) for r in rows))

    def mean(x):
        return float(np.nanmean(x)) if n and not np.all(np.isnan(x)) else float("nan")

    # 每批的墙钟只计一次（同一批的回合共享 batch_wall_s）
    batch_wall = {}
    for r in rows:
        batch_wall[int(_f(r["batch_idx"]))] = _f(r["batch_wall_s"])
    total_wall = float(sum(batch_wall.values())) if batch_wall else float("nan")
    return {
        "n_episodes": n,
        "n_success": n_success,
        "success_rate": n_success / n if n else float("nan"),
        "success_ci95_wilson": [lo, hi],
        "mean_policy_calls": mean(calls),
        "mean_smoothness": mean(smooth),
        "mean_steps": mean(steps),
        "mean_steps_success": mean(np.array([s for s, ok in zip(steps, succ) if ok] or [np.nan])),
        "mean_sum_reward": mean(np.array([_f(r["sum_reward"]) for r in rows])),
        "mean_max_reward": mean(np.array([_f(r["max_reward"]) for r in rows])),
        "n_capped": capped,
        "n_batches": len(batch_wall),
        "total_batch_wall_s": total_wall,
        "wall_s_per_episode": total_wall / n if n else float("nan"),
    }


def read_episodes(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_csv(csv_path: Path) -> dict:
    return summarize_rows(read_episodes(csv_path))
