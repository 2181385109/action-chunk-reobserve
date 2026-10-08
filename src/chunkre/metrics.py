"""纯计算：平滑度、Wilson 置信区间、从逐回合行重新算汇总。不依赖仿真，单元测试覆盖。"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .adaptive import ADAPTIVE_COLUMNS
from .perturb import PERTURB_COLUMNS, PUSH_STATUS

# 逐回合 CSV 的列（顺序固定）。TIME_COLUMNS 是墙钟类列，复现性比较时排除。
EPISODE_COLUMNS = [
    "batch_idx", "env_idx", "seed", "model", "mode", "k",
    "success", "steps", "max_steps", "capped",
    "sum_reward", "max_reward", "n_policy_calls", "smoothness",
    "wall_s", "batch_wall_s", "batch_policy_s", "batch_env_s",
    "path_length", "jerk", "reversal_rate",
    *PERTURB_COLUMNS,     # T4 扰动列（chunkre.perturb.PERTURB_COLUMNS），排在最后
]
TIME_COLUMNS = ["wall_s", "batch_wall_s", "batch_policy_s", "batch_env_s"]
# T3 新增的逐回合指标列（排在最后）。T2 写出的旧 CSV 没有这些列：续跑时沿用旧表头，汇总时跳过。
MOTION_COLUMNS = ["path_length", "jerk", "reversal_rate"]
# reversal_rate 的“静止”阈值（环境动作单位：关节为弧度，夹爪为归一化开合度 0..1）
REVERSAL_EPS = 1e-5


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


def _as_2d(actions) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float64)
    return a[:, None] if a.ndim == 1 else a


def path_length(actions) -> float:
    """路径长度 = 相邻两步动作差的 L2 范数的总和（手臂按动作目标“总共走了多远”）。

    actions: (T, action_dim)，实际发给环境的动作（环境单位）。T < 2 时返回 0.0（没有移动）。
    """
    a = _as_2d(actions)
    if a.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())


def jerk(actions) -> float:
    """jerk = 动作二阶差分的 L2 范数的平均值：mean_t ||a_t - 2 a_{t-1} + a_{t-2}||_2。

    动作是关节目标位置，所以二阶差分其实是“目标加速度”；这里按 T3 提示词的定义叫 jerk。
    匀速运动为 0，忽快忽慢或来回抖动时变大。T < 3 时返回 nan。
    """
    a = _as_2d(actions)
    if a.shape[0] < 3:
        return float("nan")
    return float(np.linalg.norm(np.diff(a, n=2, axis=0), axis=1).mean())


def reversal_rate(actions, eps: float = REVERSAL_EPS) -> float:
    """方向反转率：各关节相邻两步运动方向相反的比例。

    对每个关节 j 和每个 t，看相邻两步的变化量 d_{t-1,j} = a_{t,j} - a_{t-1,j} 和 d_{t,j}；
    两者绝对值都 > eps（默认 REVERSAL_EPS = 1e-5，关节为弧度、夹爪为归一化开合度，约等于“没动”）
    时这一对才计入分母，符号相反计为一次反转。返回 反转次数 / 计入的对数（所有关节合计）。
    任一步变化量 <= eps 的对被忽略（静止 → 再起步不算反转）。没有任何计入的对（如完全不动、T < 3）时返回 nan。
    匀速运动为 0；每步来回抖动为 1。
    """
    a = _as_2d(actions)
    if a.shape[0] < 3:
        return float("nan")
    d = np.diff(a, axis=0)
    prev, cur = d[:-1], d[1:]
    valid = (np.abs(prev) > eps) & (np.abs(cur) > eps)
    n = int(valid.sum())
    if n == 0:
        return float("nan")
    return float((valid & (np.sign(prev) != np.sign(cur))).sum() / n)


def motion_metrics(actions) -> dict:
    """一回合的三个运动指标（写进 episodes.csv 的 MOTION_COLUMNS）。"""
    return dict(path_length=path_length(actions), jerk=jerk(actions), reversal_rate=reversal_rate(actions))


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
    # T3 运动指标：只有所有行都有该列时才汇总（T2 旧 CSV 没有 → 汇总里不出现这些键，与旧 summary.json 完全一致）
    motion = {}
    for c in MOTION_COLUMNS:
        if n and all(c in r for r in rows):
            motion[f"mean_{c}"] = mean(np.array([_f(r[c]) for r in rows]))
    # T4 扰动：只有 CSV 有扰动列且确实开了扰动时才汇总（无扰动 / 旧 CSV 的 summary 键不变）
    pert = {}
    if n and all("perturb_kind" in r for r in rows):
        kinds = sorted({r["perturb_kind"] for r in rows} - {"", "none"})
        if kinds:
            pert["perturb_kind"] = kinds[0] if len(kinds) == 1 else kinds
            strengths = sorted({_f(r["perturb_strength"]) for r in rows})
            pert["perturb_strength"] = strengths[0] if len(strengths) == 1 else strengths
            if "push" in kinds:
                st = [r.get("push_status", "") for r in rows]
                pert["push_status_counts"] = {k: st.count(k) for k in PUSH_STATUS}
                pert["n_push_clipped"] = sum(1 for r in rows if r.get("push_clip"))
                pushed = [_f(r["push_d_actual"]) for r in rows if r.get("push_status") == "pushed"]
                pert["mean_push_d_actual"] = float(np.mean(pushed)) if pushed else float("nan")
    # T11a adaptive：只有 CSV 有这些列（mode=adaptive）时才汇总（其他模式的 summary 键不变）
    adapt = {}
    if n and all(c in r for r in rows for c in ADAPTIVE_COLUMNS):
        for c in ADAPTIVE_COLUMNS:
            adapt[f"mean_{c}"] = mean(np.array([_f(r[c]) for r in rows]))
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
        **motion,
        **pert,
        **adapt,
    }


def csv_columns(csv_path: Path) -> list[str] | None:
    """已有 CSV 的表头（文件不存在或为空时 None）。续跑旧格式 CSV 时沿用它。"""
    if not Path(csv_path).exists():
        return None
    with open(csv_path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), None)
    return header or None


def read_episodes(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_csv(csv_path: Path) -> dict:
    return summarize_rows(read_episodes(csv_path))
