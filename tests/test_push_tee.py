"""T8b PushT 推 T 形块的单元测试（不跑仿真：T 形块 / 智能体 / 墙用假的 TeeSim 代替，几何取 gym_pusht 的真实尺寸）。"""
import math
import random

import numpy as np
import pytest

from chunkre.config import EvalConfig
from chunkre.perturb import (PERTURB_COLUMNS, PUSH_STATUS, PushCube, PushTee, episode_rng, make_perturbation,
                             point_convex_polygon_distance, polys_at, tee_agent_gap, tee_origin_rect)

# gym_pusht add_tee(scale=30, length=4) 的两个矩形（局部坐标，body 原点 = 横条下边中点）
TEE_LOCAL = [np.array([(-60, 30), (60, 30), (60, 0), (-60, 0)], float),
             np.array([(-15, 30), (-15, 120), (15, 120), (15, 30)], float)]
WORKSPACE = (7.0, 504.0, 7.0, 504.0)     # 墙线段在 5 / 506，半径 2
AGENT_R = 15.0


class FakeTeeSim:
    """与 TeeSim 接口相同：pose / agent_xy / world_polys / set_xy + local_polys / workspace / agent_radius。"""

    def __init__(self, xy=(256.0, 256.0), angle=0.3, agent=(60.0, 60.0)):
        self.local_polys = TEE_LOCAL
        self.workspace = WORKSPACE
        self.agent_radius = AGENT_R
        self.p = np.array([xy[0], xy[1], angle], dtype=np.float64)
        self.vel = np.array([3.0, -2.0, 0.5])      # vx, vy, 角速度
        self.agent = np.array(agent, dtype=np.float64)
        self.n_set = 0

    def pose(self):
        return self.p.copy()

    def agent_xy(self):
        return self.agent.copy()

    def world_polys(self, xy=None):
        return polys_at(self.local_polys, self.p[2], self.p[:2] if xy is None else xy)

    def set_xy(self, xy):
        self.p[:2] = xy
        self.vel[:] = 0.0
        self.n_set += 1


class FakeEnv:
    def __init__(self, sims, seeds=None):
        self.sims = sims
        self.seeds = seeds if seeds is not None else list(range(len(sims)))


def fake_factory(env, i):
    return env.sims[i]


def make_push(d, **params):
    return PushTee(d, params, sim_factory=fake_factory)


def run_episode(p, env, alive_until=None, T=300):
    B = len(env.sims)
    p.begin_batch(env, env.seeds)
    for t in range(T):
        alive = np.array([alive_until is None or t < alive_until[i] for i in range(B)])
        p.on_obs(t, {}, alive)
        p.on_action(t, np.zeros((B, 2)), alive)
        p.on_state(t, env, alive)
    p.end_batch(np.full(B, T))


def inside_workspace(polys, tol=1e-9):
    v = np.vstack(polys)
    x0, x1, y0, y1 = WORKSPACE
    return v[:, 0].min() >= x0 - tol and v[:, 0].max() <= x1 + tol and v[:, 1].min() >= y0 - tol \
        and v[:, 1].max() <= y1 + tol


# ----------------------------------------------------------------------------------------------
# 随机数
# ----------------------------------------------------------------------------------------------
def test_push_tee_rng_does_not_touch_global_state():
    import torch

    random.seed(11)
    np.random.seed(22)
    torch.manual_seed(33)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    env = FakeEnv([FakeTeeSim() for _ in range(20)])
    p = make_push(60.0)
    run_episode(p, env)
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    assert before[0] == after[0]
    assert all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(before[1], after[1]))
    assert torch.equal(before[2], after[2])
    x = (random.random(), np.random.rand(), torch.rand(1).item())
    random.seed(11)
    np.random.seed(22)
    torch.manual_seed(33)
    assert x == (random.random(), np.random.rand(), torch.rand(1).item())


def test_push_tee_draws_match_episode_rng_and_are_order_independent():
    seeds = [0, 5, 77, 5001, 9999]
    a, b = make_push(20.0), make_push(80.0)
    a.begin_batch(None, seeds)
    b.begin_batch(None, seeds[::-1])
    assert a.t_push == b.t_push[::-1] and a.theta == b.theta[::-1]
    for s, t, th in zip(seeds, a.t_push, a.theta):
        g = episode_rng(s, "push")
        assert t == int(g.integers(PushTee.DEFAULTS["t_lo"], PushTee.DEFAULTS["t_hi"] + 1))
        assert th == float(g.uniform(0.0, 2.0 * math.pi))
    assert all(PushTee.DEFAULTS["t_lo"] <= t <= PushTee.DEFAULTS["t_hi"] for t in a.t_push)


# ----------------------------------------------------------------------------------------------
# 几何
# ----------------------------------------------------------------------------------------------
def test_point_polygon_distance():
    sq = np.array([(0, 0), (10, 0), (10, 10), (0, 10)], float)
    assert point_convex_polygon_distance((5, 5), sq) == 0.0
    assert point_convex_polygon_distance((10, 5), sq) == 0.0           # 边上
    assert point_convex_polygon_distance((13, 5), sq) == pytest.approx(3.0)
    assert point_convex_polygon_distance((13, 14), sq) == pytest.approx(5.0)   # 角点
    assert point_convex_polygon_distance((13, 14), sq[::-1]) == pytest.approx(5.0)   # 顶点顺序无关


def test_tee_agent_gap_uses_both_rectangles():
    polys = polys_at(TEE_LOCAL, 0.0, (256, 256))          # 横条 y∈[256,286]，竖条 x∈[241,271]、y∈[286,376]
    assert tee_agent_gap(polys, (256, 241), AGENT_R) == pytest.approx(0.0)       # 正好贴着横条下边
    assert tee_agent_gap(polys, (256, 240), AGENT_R) == pytest.approx(1.0)
    assert tee_agent_gap(polys, (290, 330), AGENT_R) == pytest.approx(4.0)       # 竖条右边 x=271
    assert tee_agent_gap(polys, (256, 330), AGENT_R) == pytest.approx(-AGENT_R)  # 圆心在竖条里


def test_tee_origin_rect_keeps_whole_tee_inside():
    polys = polys_at(TEE_LOCAL, 0.0, (256, 256))
    r = tee_origin_rect(polys, (256, 256), WORKSPACE)
    assert r == pytest.approx((67.0, 444.0, 7.0, 384.0))       # x：±60；y：0..120
    # 起点已经压进墙 0.5（穿透）：当前位置算进允许范围
    polys2 = polys_at(TEE_LOCAL, 0.0, (66.5, 256))
    r2 = tee_origin_rect(polys2, (66.5, 256), WORKSPACE)
    assert r2[0] == pytest.approx(66.5) and r2[1] == pytest.approx(444.0)


# ----------------------------------------------------------------------------------------------
# 推的逻辑
# ----------------------------------------------------------------------------------------------
def test_push_tee_moves_exactly_d_no_rotation_and_zeroes_velocity():
    sims = [FakeTeeSim() for _ in range(20)]
    env = FakeEnv(sims)
    p = make_push(40.0)
    run_episode(p, env)
    for i, sim in enumerate(sims):
        r = p.episode_info(i)
        assert r["push_status"] == "pushed" and r["push_clip"] == ""
        pre = np.array([r["cube_pre_x"], r["cube_pre_y"]])
        post = np.array([r["cube_post_x"], r["cube_post_y"]])
        assert np.linalg.norm(post - pre) == pytest.approx(40.0, abs=1e-9)
        th = math.radians(r["push_angle_deg"])
        assert np.allclose((post - pre) / 40.0, [math.cos(th), math.sin(th)])
        assert sim.p[2] == 0.3                       # 角度不变
        assert np.all(sim.vel == 0) and sim.n_set == 1
        assert r["push_d_actual"] == pytest.approx(40.0)
        assert r["cube_pre_z"] == "" and r["cube_post_z"] == ""


def test_push_tee_zero_strength_never_writes_state():
    sims = [FakeTeeSim() for _ in range(5)]
    p = make_push(0.0)
    run_episode(p, FakeEnv(sims))
    for i, sim in enumerate(sims):
        r = p.episode_info(i)
        assert r["push_status"] == "zero" and r["push_d_actual"] == 0.0 and r["push_clip"] == ""
        assert sim.n_set == 0 and np.all(sim.vel == [3.0, -2.0, 0.5])
        assert (r["cube_pre_x"], r["cube_pre_y"]) == (r["cube_post_x"], r["cube_post_y"])


def test_push_tee_wall_clip_keeps_direction_and_tee_inside():
    # 角度 0，body 在 (430, 370)：x 方向只剩 14、y 方向只剩 14 的空间 -> 大多数方向都会截断
    sims = [FakeTeeSim(xy=(430.0, 370.0), angle=0.0, agent=(30.0, 30.0)) for _ in range(40)]
    env = FakeEnv(sims)
    p = make_push(80.0)
    run_episode(p, env)
    n_clip = 0
    for i, sim in enumerate(sims):
        r = p.episode_info(i)
        assert r["push_status"] == "pushed"
        assert inside_workspace(sim.world_polys())
        disp = np.array([r["cube_post_x"] - 430.0, r["cube_post_y"] - 370.0])
        assert np.linalg.norm(disp) == pytest.approx(r["push_d_actual"], abs=1e-9)
        if r["push_clip"]:
            n_clip += 1
            assert r["push_clip"] == "wall" and r["push_d_actual"] < 80.0
            if r["push_d_actual"] > 1e-9:
                th = math.radians(r["push_angle_deg"])
                assert np.allclose(disp / np.linalg.norm(disp), [math.cos(th), math.sin(th)])
                # 截断后正好贴到墙（某条边界达到）
                v = np.vstack(sim.world_polys())
                assert min(v[:, 0].min() - 7, 504 - v[:, 0].max(), v[:, 1].min() - 7, 504 - v[:, 1].max()) \
                    == pytest.approx(0.0, abs=1e-9)
        else:
            assert r["push_d_actual"] == pytest.approx(80.0)
    assert 0 < n_clip < 40


def test_push_tee_skips_when_new_position_overlaps_agent():
    # T 在 (256,256) 角度 0；智能体在横条右侧，间隙 5：往 +x 推 40 会重叠 -> skipped_overlap；往 -x 推 -> pushed
    def run_dir(theta_deg, agent):
        sim = FakeTeeSim(xy=(256.0, 256.0), angle=0.0, agent=agent)
        p = make_push(40.0, max_redraws=0)       # T8b 行为：不重抽，重叠直接跳过
        p.begin_batch(FakeEnv([sim]), [0])
        p.theta = [math.radians(theta_deg)]
        p.on_state(p.t_push[0], FakeEnv([sim]), np.array([True]))
        return p.episode_info(0), sim

    r, sim = run_dir(0.0, agent=(256 + 60 + 15 + 5, 271))      # 初始间隙 5，推 40 后重叠
    assert r["push_status"] == "skipped_overlap" and r["push_d_actual"] == 0.0
    assert sim.n_set == 0 and np.all(sim.vel == [3.0, -2.0, 0.5])
    assert (r["cube_pre_x"], r["cube_pre_y"]) == (r["cube_post_x"], r["cube_post_y"])
    r, sim = run_dir(180.0, agent=(256 + 60 + 15 + 5, 271))    # 反方向：离开智能体
    assert r["push_status"] == "pushed" and sim.n_set == 1
    r, sim = run_dir(0.0, agent=(256 + 60 + 15 + 41, 271))     # 间隙 41 > 40：推完还差 1，不重叠
    assert r["push_status"] == "pushed"
    assert tee_agent_gap(sim.world_polys(), sim.agent_xy(), AGENT_R) == pytest.approx(1.0)
    # 只看落点：T 整个越过智能体（推 200，智能体在起点和终点之间）但落点不重叠 -> 照推
    sim = FakeTeeSim(xy=(150.0, 256.0), angle=0.0, agent=(240.0, 271.0))
    p = make_push(200.0, max_redraws=0)
    p.begin_batch(FakeEnv([sim]), [0])
    p.theta = [0.0]
    p.on_state(p.t_push[0], FakeEnv([sim]), np.array([True]))
    assert p.episode_info(0)["push_status"] == "pushed" and p.episode_info(0)["push_d_actual"] == pytest.approx(200.0)


def test_push_tee_overlap_checked_after_wall_clip():
    # 往 +x 推 80 会撞墙截断到 14；智能体在截断后位置之外 -> pushed（若用未截断位置会误判为重叠）
    sim = FakeTeeSim(xy=(430.0, 200.0), angle=0.0, agent=(497.0, 100.0))   # 智能体离 T 很远
    p = make_push(80.0)
    env = FakeEnv([sim])
    p.begin_batch(env, [0])
    p.theta = [0.0]
    p.on_state(p.t_push[0], env, np.array([True]))
    r = p.episode_info(0)
    assert r["push_clip"] == "wall" and r["push_d_actual"] == pytest.approx(14.0)
    assert r["push_status"] == "pushed"
    # 推前间隙 0.6（不重叠），截断后的位置与智能体重叠 -> 跳过，且仍记录截断
    sim2 = FakeTeeSim(xy=(430.0, 200.0), angle=0.0, agent=(502.0, 240.0))
    assert tee_agent_gap(sim2.world_polys(), sim2.agent_xy(), AGENT_R) > 0
    p2 = make_push(80.0, max_redraws=0)
    env2 = FakeEnv([sim2])
    p2.begin_batch(env2, [0])
    p2.theta = [0.0]
    p2.on_state(p2.t_push[0], env2, np.array([True]))
    r2 = p2.episode_info(0)
    assert r2["push_status"] == "skipped_overlap" and r2["push_clip"] == "wall" and sim2.n_set == 0


def test_push_tee_not_reached_and_exactly_once():
    sims = [FakeTeeSim(), FakeTeeSim()]
    env = FakeEnv(sims, seeds=[3, 4])
    p = make_push(30.0, t_lo=40, t_hi=50)
    run_episode(p, env, alive_until=[5, 300])
    assert p.episode_info(0)["push_status"] == "not_reached" and sims[0].n_set == 0
    assert p.episode_info(1)["push_status"] == "pushed" and sims[1].n_set == 1
    assert set(p.episode_info(0)) <= set(PERTURB_COLUMNS)
    assert set(p.episode_info(1)) == set(PERTURB_COLUMNS)
    assert {p.episode_info(i)["push_status"] for i in range(2)} <= set(PUSH_STATUS)


# ----------------------------------------------------------------------------------------------
# T8c：重叠时重抽方向
# ----------------------------------------------------------------------------------------------
BLOCKED_AGENT = (256.0 + 60 + 15 + 5, 271.0)     # T 在 (256,256) 角度 0：横条右端外 5 处，往右推 40 必重叠


def expected_redraws(seed, d, xy=(256.0, 256.0), angle=0.0, agent=BLOCKED_AGENT, max_redraws=20):
    """独立复现：episode_rng 先抽 push_step、θ，再依次抽重抽的 θ，直到不重叠（不含墙截断的情形）。"""
    g = episode_rng(seed, "push")
    g.integers(PushTee.DEFAULTS["t_lo"], PushTee.DEFAULTS["t_hi"] + 1)
    th = float(g.uniform(0.0, 2.0 * math.pi))
    for n in range(max_redraws + 1):
        target = np.array(xy) + d * np.array([math.cos(th), math.sin(th)])
        if tee_agent_gap(polys_at(TEE_LOCAL, angle, target), agent, AGENT_R) >= 0:
            return n, th
        if n == max_redraws:
            return n, th
        th = float(g.uniform(0.0, 2.0 * math.pi))


def run_redraw(seeds, d=40.0, **params):
    sims = [FakeTeeSim(xy=(256.0, 256.0), angle=0.0, agent=BLOCKED_AGENT) for _ in seeds]
    p = make_push(d, **params)
    run_episode(p, FakeEnv(sims, seeds=list(seeds)))
    return p, sims


def test_push_tee_redraw_is_deterministic_and_follows_episode_rng_stream():
    seeds = list(range(40))
    p1, sims = run_redraw(seeds)
    p2, _ = run_redraw(seeds[::-1])                 # 批内顺序不同
    n_redrawn = 0
    for j, s in enumerate(seeds):
        a, b = p1.episode_info(j), p2.episode_info(len(seeds) - 1 - j)
        assert a == b                                # 同一种子 -> 同样的重抽序列和结果
        n, th = expected_redraws(s, 40.0)
        assert a["push_redraws"] == n
        assert a["push_angle_deg"] == round(math.degrees(th), 6)
        assert a["push_status"] == "pushed"
        # 位移仍是 d，方向 = 最终方向，且落点不与智能体重叠
        post = np.array([a["cube_post_x"], a["cube_post_y"]])
        assert np.linalg.norm(post - 256.0) == pytest.approx(40.0, abs=1e-9)
        assert tee_agent_gap(sims[j].world_polys(), sims[j].agent_xy(), AGENT_R) >= 0
        n_redrawn += n > 0
    assert 0 < n_redrawn < len(seeds)                # 有的回合重抽，有的一次就过
    # 重抽次数与 k 无关：只由种子和几何决定 -> 同样的种子再跑一次（新对象）完全一样
    p3, _ = run_redraw(seeds)
    assert [p3.episode_info(j) for j in range(len(seeds))] == [p1.episode_info(j) for j in range(len(seeds))]


def test_push_tee_redraw_cap_is_20_then_skipped():
    # d=5，智能体圆心在竖条正中：往任何方向平移 5 都还重叠 -> 用满 20 次重抽后 skipped_overlap
    def blocked(seed, **params):
        sim = FakeTeeSim(xy=(256.0, 256.0), angle=0.0, agent=(256.0, 330.0))
        p = make_push(5.0, **params)
        run_episode(p, FakeEnv([sim], seeds=[seed]))
        return p.episode_info(0), sim

    r, sim = blocked(7)
    assert r["push_status"] == "skipped_overlap" and r["push_redraws"] == 20 and r["push_d_actual"] == 0.0
    assert sim.n_set == 0 and np.all(sim.vel == [3.0, -2.0, 0.5])
    assert (r["cube_pre_x"], r["cube_pre_y"]) == (r["cube_post_x"], r["cube_post_y"])
    g = episode_rng(7, "push")
    g.integers(20, 81)
    th = float(g.uniform(0.0, 2.0 * math.pi))
    for _ in range(20):
        th = float(g.uniform(0.0, 2.0 * math.pi))     # 第 20 次重抽的方向 = 最后试的方向
    assert r["push_angle_deg"] == round(math.degrees(th), 6)
    r, _ = blocked(7, max_redraws=3)
    assert r["push_status"] == "skipped_overlap" and r["push_redraws"] == 3
    r, _ = blocked(7, max_redraws=0)
    assert r["push_status"] == "skipped_overlap" and r["push_redraws"] == 0


def test_push_tee_redraw_does_not_touch_global_random_state():
    import torch

    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    p, _ = run_redraw(list(range(30)))
    assert max(p.episode_info(j)["push_redraws"] for j in range(30)) > 0     # 确实发生了重抽
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    assert before[0] == after[0]
    assert all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(before[1], after[1]))
    assert torch.equal(before[2], after[2])


def test_push_tee_redraws_zero_when_no_overlap_and_blank_when_not_reached():
    sims = [FakeTeeSim(), FakeTeeSim(), FakeTeeSim()]
    p = make_push(0.0)
    run_episode(p, FakeEnv(sims))
    assert [p.episode_info(i)["push_redraws"] for i in range(3)] == [0, 0, 0]       # d=0：zero，不重抽
    p = make_push(40.0)
    run_episode(p, FakeEnv([FakeTeeSim(), FakeTeeSim()], seeds=[3, 4]), alive_until=[5, 300])
    assert p.episode_info(0)["push_redraws"] == ""                                  # not_reached
    assert p.episode_info(1)["push_redraws"] == 0                                   # 智能体远，首个方向就过


# ----------------------------------------------------------------------------------------------
# 配置 / 分派
# ----------------------------------------------------------------------------------------------
def test_push_dispatch_by_model():
    assert isinstance(make_perturbation({"kind": "push", "strength": 40}, "dp"), PushTee)
    assert isinstance(make_perturbation({"kind": "push", "strength": 0.04}, "act"), PushCube)
    assert isinstance(make_perturbation({"kind": "push", "strength": 0.04}), PushCube)   # 旧调用方式不变
    c = EvalConfig(model="dp", seeds="0-9", perturb={"kind": "push", "strength": 40})
    assert c.perturb["params"] == PushTee.DEFAULTS
    with pytest.raises(ValueError):
        EvalConfig(model="dp", seeds="0-9", perturb={"kind": "push", "strength": 40, "params": {"lift_thresh": 1}})
    with pytest.raises(ValueError):
        make_perturbation({"kind": "push", "strength": 40, "params": {"t_lo": 9, "t_hi": 3}}, "dp")
