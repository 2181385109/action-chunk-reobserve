"""T13 失败案例分析（探索性，只描述）：纯计算，不依赖仿真 / torch，单元测试覆盖。

阈值和定义写在 results/failure/definitions.md（在看任何种子 7000–7049 的数据之前单独提交，之后不改），
数值由调试种子 0–9 的标定数据（results/failure_calib/，scripts/failure_calib.sh）定出，这里是同一组常量。

aloha 14 维动作 / 关节：0–5 左臂关节、6 左夹爪、7–12 右臂关节、13 右夹爪（夹爪为归一化开合度，1 张开、0 闭合）。
轨迹（chunkre.traj）记录的是“执行第 t 步动作之前”的状态，与第 t 步动作配对。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIPPERS = [6, 13]

# ---- 失败分类（definitions.md §2） ----
REST_STEPS = (10, 20)    # 方块初始放在 z=0.05 m、开局几步落到桌面；“初始高度” = 第 10–19 步方块 z 的中位数
H_LIFT = 0.04            # 抬起：方块 z 曾超过 初始高度 + H_LIFT（米）；理由见 definitions.md §2
CLASSES = ("success", "never_lifted", "lifted_failed")
CLASS_LABEL = {"success": "success", "never_lifted": "never lifted", "lifted_failed": "lifted, not completed"}

# ---- 推之后的“闭合夹爪”（definitions.md §3） ----
GRASP_ARM = "right"      # 先抓方块的手臂（标定数据：成功回合里先抬起方块的都是右臂）
GRIP_DIM = {"left": 6, "right": 13}
TIP_IDX = {"left": (0, 1), "right": (2, 3)}   # traj finger_tip 的第 3 维：左臂左指、左臂右指、右臂左指、右臂右指
GRIP_OPEN = 0.8          # 夹爪指令（动作第 13 维）先 >= GRIP_OPEN（张开） ...
GRIP_CLOSE = 0.5         # ... 之后第一次 < GRIP_CLOSE 的步 = 闭合时刻


def rest_height(cube_z: np.ndarray) -> float:
    """初始高度：第 REST_STEPS 区间内方块 z 的中位数（回合不够长时用已有的最后几步）。"""
    z = np.asarray(cube_z, dtype=np.float64)
    z = z[~np.isnan(z)]
    lo, hi = REST_STEPS
    seg = z[lo:hi] if len(z) >= hi else z[-min(len(z), hi - lo):]
    return float(np.median(seg))


def max_lift(cube_z: np.ndarray) -> float:
    """第 REST_STEPS[0] 步起方块最高高度 − 初始高度（开局下落那几步不算，否则 z=0.05 的初始位置会被当成抬起）。"""
    z = np.asarray(cube_z, dtype=np.float64)
    tail = z[REST_STEPS[0]:] if len(z) > REST_STEPS[0] else z
    return float(np.nanmax(tail) - rest_height(z))


def classify_episode(success: bool, cube_z: np.ndarray, h_lift: float = H_LIFT) -> str:
    if success:
        return "success"
    return "lifted_failed" if max_lift(cube_z) > h_lift else "never_lifted"


def first_repredict_after(repredict: np.ndarray, t0: int, steps: int) -> int | None:
    """第一次重新预测的步 t（t0 < t < steps），返回 t − t0；回合内没有则 None。
    推发生在第 t0 步策略选好动作之后，所以 t == t0 那次用的是推之前的画面，不算。"""
    r = np.flatnonzero(np.asarray(repredict, dtype=bool)[:steps])
    r = r[r > t0]
    return int(r[0] - t0) if len(r) else None


def close_events(grip_cmd: np.ndarray, open_thr: float = GRIP_OPEN, close_thr: float = GRIP_CLOSE) -> list[int]:
    """夹爪指令序列里的“闭合”时刻：先 >= open_thr（张开）之后第一次 < close_thr 的步；之后要再张开才算下一次。"""
    g = np.asarray(grip_cmd, dtype=np.float64)
    out, armed = [], False
    for t, v in enumerate(g):
        if np.isnan(v):
            break
        if v >= open_thr:
            armed = True
        elif armed and v < close_thr:
            out.append(t)
            armed = False
    return out


def first_close_after(grip_cmd: np.ndarray, t0: int) -> int | None:
    """推之后（t > t0）第一次闭合的步；闭合必须是在推之后才完成的（张开可以在推之前）。"""
    ev = [t for t in close_events(grip_cmd) if t > t0]
    return ev[0] if ev else None


def grip_point(finger_tip: np.ndarray, arm: str = GRASP_ARM) -> np.ndarray:
    """某只手臂两个指尖的中点。finger_tip: (T, 4, 3) -> (T, 3)。"""
    a, b = TIP_IDX[arm]
    return 0.5 * (np.asarray(finger_tip)[:, a] + np.asarray(finger_tip)[:, b])


def horiz_dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(p, dtype=np.float64)[:2] - np.asarray(q, dtype=np.float64)[:2]))


def switch_jumps(action: np.ndarray, repredict: np.ndarray, steps: int, delay: int = 0) -> dict:
    """换块时刻的动作跳变。新块在第 t_r 步预测（t_r > 0），在第 t_e = t_r + delay 步第一次被执行；
    跳变 = action[t_e] − action[t_e − 1]（14 维 L2；12 个关节 L2；2 个夹爪 L2）。t_e >= steps 的不算。
    另给出非换块步（块内相邻两步）的同样三种差，作参照。"""
    a = np.asarray(action, dtype=np.float64)[:steps]
    rp = np.flatnonzero(np.asarray(repredict, dtype=bool)[:steps])
    te = np.array([t + delay for t in rp if t > 0 and t + delay < steps], dtype=int)
    d = np.diff(a, axis=0)                         # d[t-1] = a[t] - a[t-1]
    full, arm, grip = (np.linalg.norm(d, axis=1), np.linalg.norm(d[:, ARM], axis=1),
                       np.linalg.norm(d[:, GRIPPERS], axis=1))
    is_sw = np.zeros(len(d), dtype=bool)
    is_sw[te - 1] = True
    first_exec = delay if delay > 0 else 0         # 延迟时第 delay 步才开始执行策略输出（之前是保持位置）
    base = ~is_sw
    base[:max(first_exec, 0)] = False              # 去掉保持阶段和从保持切到第一个策略动作的那一步
    return dict(t_exec=te, full=full[te - 1], arm=arm[te - 1], grip=grip[te - 1],
                base_full=full[base], base_arm=arm[base], base_grip=grip[base])


BACK_WINDOW = 5   # “往回跳”：跳变方向与跳变前 BACK_WINDOW 步的运动方向比较


def jump_backtrack_cos(action: np.ndarray, t_exec, window: int = BACK_WINDOW) -> np.ndarray:
    """每个换块执行步 t_e：12 关节跳变 a[t_e] − a[t_e−1] 与之前运动 a[t_e−1] − a[t_e−1−window] 的余弦（< 0 = 往回跳）。
    之前不够 window 步或任一向量范数 < 1e-6 时为 nan。"""
    a = np.asarray(action, dtype=np.float64)[:, ARM]
    out = []
    for t in np.asarray(t_exec, dtype=int):
        if t - 1 - window < 0:
            out.append(np.nan)
            continue
        j, v = a[t] - a[t - 1], a[t - 1] - a[t - 1 - window]
        nj, nv = np.linalg.norm(j), np.linalg.norm(v)
        out.append(float(j @ v / (nj * nv)) if nj > 1e-6 and nv > 1e-6 else np.nan)
    return np.asarray(out, dtype=np.float64)


def load_traj_episodes(cell_dir: Path) -> dict[int, dict]:
    """读一格的 traj/batch_*.npz，按种子拆成逐回合数组（只保留存活的步）。"""
    out = {}
    for f in sorted(Path(cell_dir, "traj").glob("batch_*.npz")):
        z = np.load(f)
        for i, s in enumerate(z["seeds"]):
            n = int(z["steps"][i])
            ep = dict(steps=n, repredict=z["repredict"][:n].astype(bool))
            for key in ("action", "policy_action", "agent_pos", "cube_pos", "cube_pos_after", "finger_tip"):
                if key in z.files:
                    ep[key] = z[key][:n, i].astype(np.float64)
            out[int(s)] = ep
    return out


def describe(x) -> dict:
    """分布摘要：n、均值、中位数、四分位、p90、最大。"""
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], dtype=np.float64)
    if len(x) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, q25=np.nan, q75=np.nan, p90=np.nan, min=np.nan, max=np.nan)
    q = np.percentile(x, [25, 50, 75, 90])
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(q[1]), q25=float(q[0]), q75=float(q[2]),
                p90=float(q[3]), min=float(x.min()), max=float(x.max()))
