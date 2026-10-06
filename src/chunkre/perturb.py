"""扰动框架（T4）：在统一评估的逐步循环里注入可控扰动。

一个扰动对象有三个钩子（默认都是“原样返回 / 什么都不做”）：
  on_obs(t, obs, alive)    -> obs      改策略在第 t 步看到的观测（给 T5 观测噪声用）
  on_action(t, action, alive) -> action 改第 t 步发给环境的动作（给 T5 动作延迟用）
  on_state(t, env, alive)             在第 t 步改仿真状态（推方块用）
runner.run_batch 中第 t 步的顺序：
  观测 obs_t -> on_obs -> 策略出动作 -> on_action -> 记录（轨迹 / 指标） -> on_state -> env.step(动作)
所以 on_state 在“策略已经根据推之前的画面选好第 t 步动作之后、执行它之前”生效，策略最早在第 t+1 步看到推后的画面。
（T5 注意：轨迹和运动指标记录的是 on_action 之后、真正发给环境的动作。）

扰动随机数（见 episode_rng）：每回合一个独立的 numpy Generator，由（回合种子, 扰动类型）派生，
绝不调用 / 消耗 random、numpy、torch 的全局随机数。所以强度 0 时策略看到的观测和全局随机状态都与无扰动完全一致；
并且同一个种子在不同 k 下抽到的扰动（推的时刻、方向）相同。

本模块只依赖 numpy（不依赖 torch / lerobot），仿真通过 CubeSim 适配器访问，便于单元测试。
"""
from __future__ import annotations

import math
import zlib

import numpy as np

KINDS = ("none", "push", "obs_noise", "action_delay")
NOT_IMPLEMENTED = {"obs_noise": "T5", "action_delay": "T5"}

# 每回合写进 episodes.csv 的扰动列（排在最后）。无扰动时 perturb_kind=none、strength=0，其余列为空。
PERTURB_COLUMNS = [
    "perturb_kind", "perturb_strength",
    "push_step", "push_angle_deg", "push_status", "push_clip", "push_d_actual",
    "cube_pre_x", "cube_pre_y", "cube_pre_z", "cube_post_x", "cube_post_y", "cube_post_z",
]
# push_status 取值
PUSH_STATUS = ("pushed", "zero", "skipped_lifted", "not_reached")


def episode_rng(seed: int, kind: str) -> np.random.Generator:
    """回合种子 + 扰动类型 -> 独立的 numpy Generator。

    用 zlib.crc32（跨进程稳定；Python 的 hash() 对字符串每次启动随机化，不能用）。
    Generator 有自己的状态，不读写 numpy 全局随机状态，也不碰 random / torch。
    """
    return np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(kind.encode("utf-8"))]))


def normalize_spec(spec) -> dict:
    """配置里的 perturb 字段（None / {} / {'kind','strength','params'}）-> 规范形式。"""
    spec = dict(spec or {})
    unknown = set(spec) - {"kind", "strength", "params"}
    if unknown:
        raise ValueError(f"perturb 配置里有未知字段：{sorted(unknown)}（只认 kind / strength / params）")
    kind = str(spec.get("kind") or "none")
    if kind not in KINDS:
        raise ValueError(f"perturb.kind 必须是 {KINDS} 之一，收到 {kind!r}")
    strength = float(spec.get("strength") or 0.0)
    if not math.isfinite(strength) or strength < 0:
        raise ValueError(f"perturb.strength 必须是有限的非负数，收到 {strength}")
    params = dict(spec.get("params") or {})
    if kind == "none":
        if strength != 0 or params:
            raise ValueError("kind=none 时不能设置 strength / params")
    return dict(kind=kind, strength=strength, params=params)


def make_perturbation(spec, model: str | None = None) -> "Perturbation":
    s = normalize_spec(spec)
    if s["kind"] in NOT_IMPLEMENTED:
        raise NotImplementedError(f"扰动 {s['kind']} 只留了接口，具体实现在 {NOT_IMPLEMENTED[s['kind']]}")
    if s["kind"] == "none":
        return Perturbation()
    if s["kind"] == "push":
        if model is not None and model != "act":
            raise ValueError("push（推方块）目前只实现了 aloha（model=act）；PushT 版本留到第 8 天")
        return PushCube(s["strength"], s["params"])
    raise AssertionError(s["kind"])


# ----------------------------------------------------------------------------------------------
# 基类：无扰动
# ----------------------------------------------------------------------------------------------
class Perturbation:
    """无扰动（kind=none）。子类覆盖需要的钩子。"""

    kind = "none"
    touches_state = False       # True 时 runner 在 on_state 之后额外记录一次方块位置（轨迹 cube_pos_after）

    def __init__(self, strength: float = 0.0, params: dict | None = None):
        self.strength = float(strength)
        self.params = dict(params or {})
        self.seeds: list[int] = []

    # --- 生命周期 ---
    def begin_batch(self, env, seeds) -> None:
        """每批 env.reset 之后、第一步之前调用。子类在这里为每个回合建独立随机数并抽取扰动参数。"""
        self.seeds = [int(s) for s in seeds]

    def end_batch(self, steps) -> None:
        """每批结束时调用（steps：各回合实际步数）。"""

    # --- 三个钩子（默认原样返回 / 什么都不做）---
    def on_obs(self, t: int, obs, alive: np.ndarray):
        return obs

    def on_action(self, t: int, action: np.ndarray, alive: np.ndarray) -> np.ndarray:
        return action

    def on_state(self, t: int, env, alive: np.ndarray) -> None:
        return None

    # --- 记录 ---
    def episode_info(self, i: int) -> dict:
        """第 i 个环境这一回合的扰动记录（episodes.csv 的 PERTURB_COLUMNS）。"""
        return dict(perturb_kind=self.kind, perturb_strength=self.strength)

    def describe(self) -> dict:
        """写进 config.json 的扰动设置（含默认参数的实际取值）。"""
        return dict(kind=self.kind, strength=self.strength, params=self.params)


# ----------------------------------------------------------------------------------------------
# 推方块（aloha TransferCube）
# ----------------------------------------------------------------------------------------------
class CubeSim:
    """aloha 一个子环境里“方块 + 桌面 + 机械臂”的访问适配器（dm_control physics）。单元测试用假的替代。"""

    def __init__(self, physics, joint="red_box_joint", geom="red_box", table_geom="table",
                 left_base="vx300s_left", right_base="vx300s_right"):
        self.ph = physics
        m = physics.model
        j = m.name2id(joint, "joint")
        self.qadr = int(m.jnt_qposadr[j])
        self.vadr = int(m.jnt_dofadr[j])
        self.geom = m.name2id(geom, "geom")
        self.half = float(m.geom_size[self.geom][0])          # 方块半边长（立方体）
        cube_root = int(m.body_rootid[m.geom_bodyid[self.geom]])
        table = m.name2id(table_geom, "geom")
        table_root = int(m.body_rootid[m.geom_bodyid[table]])
        # 机械臂的几何体 = 根 body 不是世界、桌子、方块的几何体
        self.robot_geoms = {g for g in range(m.ngeom)
                            if int(m.body_rootid[m.geom_bodyid[g]]) not in (0, cube_root, table_root)}
        # 桌面范围：桌子网格顶点变换到世界坐标后的 xy 包围盒
        mid = int(m.geom_dataid[table])
        v = np.asarray(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], dtype=np.float64)
        R = np.asarray(physics.data.geom_xmat[table], dtype=np.float64).reshape(3, 3)
        w = v @ R.T + np.asarray(physics.data.geom_xpos[table], dtype=np.float64)
        self.table_xy = (float(w[:, 0].min()), float(w[:, 0].max()), float(w[:, 1].min()), float(w[:, 1].max()))
        self.table_top = float(w[:, 2].max())
        self.base_x = (float(physics.named.data.xpos[left_base][0]), float(physics.named.data.xpos[right_base][0]))

    def pos(self) -> np.ndarray:
        return np.array(self.ph.data.qpos[self.qadr:self.qadr + 3], dtype=np.float64)

    def set_xy(self, xy) -> None:
        """改方块自由关节的 xy（z 和姿态不变），6 个速度分量清零，然后 mj_forward 重新计算一次状态。"""
        self.ph.data.qpos[self.qadr:self.qadr + 2] = xy
        self.ph.data.qvel[self.vadr:self.vadr + 6] = 0.0
        self.ph.forward()

    def robot_penetration(self) -> float:
        """方块与机械臂几何体之间最深的穿透（米，>=0；没有接触为 0）。要在 set_xy（含 forward）之后调用。"""
        d = self.ph.data
        worst = 0.0
        for c in range(int(d.ncon)):
            con = d.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 == self.geom and g2 in self.robot_geoms) or (g2 == self.geom and g1 in self.robot_geoms):
                worst = max(worst, -float(con.dist))
        return worst


def aloha_cube_sim(env, i: int) -> CubeSim:
    return CubeSim(env.envs[i].unwrapped._env.physics)


def allowed_rect(sim, edge_margin: float, base_clearance: float) -> tuple[float, float, float, float]:
    """方块中心允许落在的矩形 (xmin, xmax, ymin, ymax)：桌面内缩 edge_margin，且离两侧机械臂底座的 x 至少 base_clearance。"""
    x0, x1, y0, y1 = sim.table_xy
    lb, rb = sim.base_x
    return (max(x0 + edge_margin, lb + base_clearance), min(x1 - edge_margin, rb - base_clearance),
            y0 + edge_margin, y1 - edge_margin)


def ray_clip(p, u, d: float, rect) -> float:
    """从 p 沿单位方向 u 走 d，最多能走多远仍在矩形 rect 内（保持方向，只缩短距离）。p 在矩形外时返回 0。"""
    x0, x1, y0, y1 = rect
    px, py = float(p[0]), float(p[1])
    eps = 1e-12
    if not (x0 - eps <= px <= x1 + eps and y0 - eps <= py <= y1 + eps):
        return 0.0
    s = float(d)
    if u[0] > eps:
        s = min(s, (x1 - px) / u[0])
    elif u[0] < -eps:
        s = min(s, (x0 - px) / u[0])
    if u[1] > eps:
        s = min(s, (y1 - py) / u[1])
    elif u[1] < -eps:
        s = min(s, (y0 - py) / u[1])
    return max(0.0, s)


class PushCube(Perturbation):
    """回合中途把方块水平推开距离 d = strength（米）。

    每回合（独立随机数 episode_rng(种子, "push")，先抽时刻再抽方向，顺序固定）：
      push_step ~ 均匀整数 [t_lo, t_hi]；角度 θ ~ 均匀 [0, 2π)（水平面，从 +x 轴逆时针）。
    在第 push_step 步（只对仍在进行的回合）：
      - 方块高于静止高度 + lift_thresh（已被抬起）→ 不推，记 skipped_lifted；
      - d == 0 → 不改仿真状态，记 zero（时刻和方向照样抽取并记录）；
      - 否则目标 = 推前 xy + d·(cosθ, sinθ)，沿该方向截断到允许矩形内（clip=table），
        改方块自由关节位置、速度清零、mj_forward；若与机械臂穿透 > arm_tol，沿同一方向二分缩短到不穿透（clip=arm）。
    回合在 push_step 之前就结束 → not_reached。
    """

    kind = "push"
    touches_state = True
    DEFAULTS = dict(
        t_lo=20,              # 窗口（含两端）。依据见 HANDOFF.md：T3 k=100 首次夹爪闭合最早 120 步、10% 分位 123.9 步
        t_hi=100,
        lift_thresh=0.01,     # 方块中心高于静止高度（= 桌面 + 半边长）超过 1 cm 视为已被抬起
        edge_margin=0.05,     # 方块中心离桌边至少 5 cm（含半边长 2 cm）
        base_clearance=0.12,  # 方块中心离机械臂底座（x = ±0.469）至少 12 cm
        arm_tol=0.001,        # 与机械臂几何体穿透超过 1 mm 视为“推进机械臂里”
        arm_bisect_iters=12,
    )

    def __init__(self, strength: float, params: dict | None = None, sim_factory=aloha_cube_sim):
        params = dict(params or {})
        unknown = set(params) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"push 的未知参数：{sorted(unknown)}；可用 {sorted(self.DEFAULTS)}")
        p = dict(self.DEFAULTS, **params)
        p["t_lo"], p["t_hi"], p["arm_bisect_iters"] = int(p["t_lo"]), int(p["t_hi"]), int(p["arm_bisect_iters"])
        if not (0 <= p["t_lo"] <= p["t_hi"]):
            raise ValueError(f"push 窗口必须 0 <= t_lo <= t_hi，收到 [{p['t_lo']}, {p['t_hi']}]")
        super().__init__(strength, p)
        self.sim_factory = sim_factory
        self._rect = None
        self._geom = None

    def begin_batch(self, env, seeds) -> None:
        super().begin_batch(env, seeds)
        self.t_push, self.theta, self.rec = [], [], []
        for s in self.seeds:
            rng = episode_rng(s, self.kind)
            t = int(rng.integers(self.params["t_lo"], self.params["t_hi"] + 1))
            th = float(rng.uniform(0.0, 2.0 * math.pi))
            self.t_push.append(t)
            self.theta.append(th)
            self.rec.append(dict(push_step=t, push_angle_deg=round(math.degrees(th), 6), push_status=None,
                                 push_clip="", push_d_actual=""))
        if self._geom is None and env is not None:
            sim = self.sim_factory(env, 0)
            self._geom = dict(table_xy=sim.table_xy, table_top=sim.table_top, cube_half=sim.half,
                              base_x=sim.base_x)
            self._rect = allowed_rect(sim, self.params["edge_margin"], self.params["base_clearance"])

    def on_state(self, t: int, env, alive: np.ndarray) -> None:
        for i in np.flatnonzero(alive):
            if t == self.t_push[i] and self.rec[i]["push_status"] is None:
                self._push(self.sim_factory(env, int(i)), int(i))

    def _push(self, sim, i: int) -> None:
        r = self.rec[i]
        pre = sim.pos()
        rect = self._rect if self._rect is not None else allowed_rect(
            sim, self.params["edge_margin"], self.params["base_clearance"])
        d = self.strength
        if pre[2] > sim.table_top + sim.half + self.params["lift_thresh"]:
            r.update(push_status="skipped_lifted", push_d_actual=0.0)
        elif d == 0.0:
            r.update(push_status="zero", push_d_actual=0.0)
        else:
            u = np.array([math.cos(self.theta[i]), math.sin(self.theta[i])])
            s_tab = ray_clip(pre[:2], u, d, rect)
            clips = ["table"] if s_tab < d - 1e-12 else []
            s = s_tab
            sim.set_xy(pre[:2] + s * u)
            if sim.robot_penetration() > self.params["arm_tol"]:
                clips.append("arm")
                lo, hi = 0.0, s           # lo：已知不穿透（推前位置视为可接受）
                for _ in range(self.params["arm_bisect_iters"]):
                    mid = 0.5 * (lo + hi)
                    sim.set_xy(pre[:2] + mid * u)
                    if sim.robot_penetration() > self.params["arm_tol"]:
                        hi = mid
                    else:
                        lo = mid
                s = lo
                sim.set_xy(pre[:2] + s * u)
            r.update(push_status="pushed", push_clip="+".join(clips), push_d_actual=float(s))
        post = sim.pos()
        r.update(cube_pre_x=pre[0], cube_pre_y=pre[1], cube_pre_z=pre[2],
                 cube_post_x=post[0], cube_post_y=post[1], cube_post_z=post[2])

    def end_batch(self, steps) -> None:
        for r in self.rec:
            if r["push_status"] is None:
                r["push_status"] = "not_reached"

    def episode_info(self, i: int) -> dict:
        return dict(super().episode_info(i), **self.rec[i])

    def describe(self) -> dict:
        return dict(super().describe(), allowed_rect_xy=self._rect, geometry=self._geom,
                    rng="numpy default_rng(SeedSequence([回合种子, crc32('push')]))；先抽 push_step 再抽角度",
                    timing="第 push_step 步：策略已选好该步动作之后、env.step 之前改方块位置")
