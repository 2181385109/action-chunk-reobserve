"""T4 扰动框架单元测试（不跑仿真：方块 / 桌面 / 机械臂用假的 CubeSim 代替）。"""
import math
import random

import numpy as np
import pytest

from chunkre.config import EvalConfig, load_config, merge_perturb, parse_param_kv
from chunkre.perturb import (PERTURB_COLUMNS, Perturbation, PushCube, allowed_rect, episode_rng,
                             make_perturbation, normalize_spec, ray_clip)

RECT = (-0.349, 0.349, 0.269, 0.931)     # 真实 aloha 模型算出的允许矩形（默认参数）


class FakeSim:
    """与 CubeSim 接口相同：pos / set_xy / robot_penetration + 几何属性。"""

    def __init__(self, xyz=(0.1, 0.5, 0.02), arm_x=None):
        self.qpos = np.array(list(xyz) + [1.0, 0, 0, 0], dtype=np.float64)
        self.qvel = np.full(6, 0.3)
        self.half, self.table_top = 0.02, 0.0
        self.table_xy = (-0.6096, 0.6096, 0.219, 0.981)
        self.base_x = (-0.469, 0.469)
        self.arm_x = arm_x          # 假机械臂：方块中心 x > arm_x 时穿透 (x - arm_x)
        self.n_set = 0

    def pos(self):
        return self.qpos[:3].copy()

    def set_xy(self, xy):
        self.qpos[:2] = xy
        self.qvel[:] = 0.0
        self.n_set += 1

    def robot_penetration(self):
        if self.arm_x is None:
            return 0.0
        return max(0.0, self.qpos[0] - self.arm_x)


class FakeEnv:
    def __init__(self, sims):
        self.sims = sims


def fake_factory(env, i):
    return env.sims[i]


def make_push(d, **params):
    return PushCube(d, params, sim_factory=fake_factory)


def run_episode(p, env, alive_until=None, T=400):
    """模拟 runner 的调用顺序：begin_batch -> 每步 on_obs/on_action/on_state -> end_batch。"""
    B = len(env.sims)
    p.begin_batch(env, list(range(B)) if not hasattr(env, "seeds") else env.seeds)
    for t in range(T):
        alive = np.array([alive_until is None or t < alive_until[i] for i in range(B)])
        p.on_obs(t, {}, alive)
        p.on_action(t, np.zeros((B, 14)), alive)
        p.on_state(t, env, alive)
    p.end_batch(np.full(B, T))


# ----------------------------------------------------------------------------------------------
# 随机数
# ----------------------------------------------------------------------------------------------
def _global_states():
    import torch

    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _same(a, b):
    import torch

    assert a[0] == b[0]
    assert all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(a[1], b[1]))
    assert torch.equal(a[2], b[2])


def test_perturb_rng_does_not_touch_global_state():
    random.seed(123)
    np.random.seed(456)
    import torch

    torch.manual_seed(789)
    before = _global_states()
    env = FakeEnv([FakeSim(arm_x=0.15) for _ in range(10)])
    env.seeds = list(range(10))
    p = make_push(0.10)
    run_episode(p, env)
    for s in range(50):
        episode_rng(s, "push").random(100)
    after = _global_states()
    _same(before, after)
    # 全局随机数之后的输出也与“没跑过扰动”完全相同
    x = (random.random(), np.random.rand(), torch.rand(1).item())
    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    assert x == (random.random(), np.random.rand(), torch.rand(1).item())


def test_same_seed_same_time_and_direction():
    a, b = make_push(0.05), make_push(0.10, t_lo=20, t_hi=100)   # 不同强度（不同 k 下扰动对象也一样新建）
    seeds = [0, 1, 2, 7, 1234]
    a.begin_batch(None, seeds)
    b.begin_batch(None, seeds)
    assert a.t_push == b.t_push and a.theta == b.theta
    c = make_push(0.10)
    c.begin_batch(None, seeds[::-1])          # 种子在批里的位置不影响
    assert c.t_push == b.t_push[::-1] and c.theta == b.theta[::-1]
    assert len(set(b.t_push)) > 1 and len(set(b.theta)) == len(seeds)
    assert all(20 <= t <= 100 for t in b.t_push)
    assert all(0 <= th < 2 * math.pi for th in b.theta)


def test_episode_rng_depends_on_seed_and_kind():
    x = episode_rng(5, "push").random(4)
    assert np.array_equal(x, episode_rng(5, "push").random(4))
    assert not np.array_equal(x, episode_rng(6, "push").random(4))
    assert not np.array_equal(x, episode_rng(5, "obs_noise").random(4))


def test_push_time_uniform_over_window():
    p = make_push(0.1, t_lo=20, t_hi=24)
    p.begin_batch(None, list(range(2000)))
    counts = np.bincount(p.t_push, minlength=25)[20:]
    assert counts.sum() == 2000 and counts.min() > 330      # 5 个值各约 400


# ----------------------------------------------------------------------------------------------
# 推方块逻辑
# ----------------------------------------------------------------------------------------------
def test_push_moves_exactly_d_and_zeroes_velocity():
    env = FakeEnv([FakeSim() for _ in range(5)])
    p = make_push(0.05)
    run_episode(p, env)
    for i, sim in enumerate(env.sims):
        r = p.episode_info(i)
        assert r["push_status"] == "pushed" and r["push_clip"] == ""
        pre = np.array([r["cube_pre_x"], r["cube_pre_y"]])
        post = np.array([r["cube_post_x"], r["cube_post_y"]])
        assert np.linalg.norm(post - pre) == pytest.approx(0.05, abs=1e-12)
        th = math.radians(r["push_angle_deg"])
        assert np.allclose((post - pre) / 0.05, [math.cos(th), math.sin(th)])
        assert r["cube_post_z"] == r["cube_pre_z"]
        assert np.all(sim.qvel == 0) and sim.n_set == 1
        assert r["push_d_actual"] == pytest.approx(0.05)


def test_zero_strength_never_writes_state():
    env = FakeEnv([FakeSim() for _ in range(5)])
    p = make_push(0.0)
    run_episode(p, env)
    for i, sim in enumerate(env.sims):
        r = p.episode_info(i)
        assert r["push_status"] == "zero" and r["push_d_actual"] == 0.0
        assert sim.n_set == 0 and np.all(sim.qvel == 0.3)
        assert (r["cube_pre_x"], r["cube_pre_y"]) == (r["cube_post_x"], r["cube_post_y"])


def test_lifted_cube_is_skipped():
    env = FakeEnv([FakeSim(xyz=(0.1, 0.5, 0.031)), FakeSim(xyz=(0.1, 0.5, 0.029))])
    p = make_push(0.05)
    run_episode(p, env)
    assert p.episode_info(0)["push_status"] == "skipped_lifted" and env.sims[0].n_set == 0
    assert p.episode_info(1)["push_status"] == "pushed"


def test_ended_env_not_pushed_and_not_reached():
    env = FakeEnv([FakeSim(), FakeSim()])
    p = make_push(0.05, t_lo=50, t_hi=60)
    run_episode(p, env, alive_until=[10, 400])     # 环境 0 第 10 步就结束了
    assert p.episode_info(0)["push_status"] == "not_reached" and env.sims[0].n_set == 0
    assert p.episode_info(1)["push_status"] == "pushed"


def test_push_happens_exactly_once_at_push_step():
    sim = FakeSim()
    env = FakeEnv([sim])
    p = make_push(0.05)
    p.begin_batch(env, [3])
    t0 = p.t_push[0]
    for t in range(400):
        before = sim.pos()
        p.on_state(t, env, np.array([True]))
        moved = not np.array_equal(before, sim.pos())
        assert moved == (t == t0)


def test_ray_clip():
    u = np.array([1.0, 0.0])
    assert ray_clip([0.0, 0.5], u, 0.1, RECT) == pytest.approx(0.1)
    assert ray_clip([0.3, 0.5], u, 0.1, RECT) == pytest.approx(0.049)
    assert ray_clip([0.3, 0.5], -u, 0.1, RECT) == pytest.approx(0.1)
    diag = np.array([1.0, 1.0]) / math.sqrt(2)
    s = ray_clip([0.3, 0.9], diag, 0.1, RECT)          # y 先到边：(0.931-0.9)/0.7071
    assert s == pytest.approx(0.031 * math.sqrt(2))
    assert ray_clip([0.5, 0.5], -u, 0.1, RECT) == 0.0   # 起点已在矩形外 -> 不动


def test_allowed_rect_from_geometry():
    r = allowed_rect(FakeSim(), edge_margin=0.05, base_clearance=0.12)
    assert r == pytest.approx(RECT)


def test_table_clip_recorded_and_direction_kept():
    sims = [FakeSim(xyz=(0.34, 0.92, 0.02)) for _ in range(20)]   # 靠近右上角：大多数方向都会截断
    env = FakeEnv(sims)
    p = make_push(0.10)
    run_episode(p, env)
    n_clip = 0
    for i, sim in enumerate(sims):
        r = p.episode_info(i)
        post = np.array([r["cube_post_x"], r["cube_post_y"]])
        assert RECT[0] - 1e-9 <= post[0] <= RECT[1] + 1e-9 and RECT[2] - 1e-9 <= post[1] <= RECT[3] + 1e-9
        disp = post - np.array([0.34, 0.92])
        assert np.linalg.norm(disp) == pytest.approx(r["push_d_actual"], abs=1e-12)
        if r["push_clip"]:
            n_clip += 1
            assert r["push_clip"] == "table" and r["push_d_actual"] < 0.10
            if r["push_d_actual"] > 1e-9:
                th = math.radians(r["push_angle_deg"])
                assert np.allclose(disp / np.linalg.norm(disp), [math.cos(th), math.sin(th)])
        else:
            assert r["push_d_actual"] == pytest.approx(0.10)
    assert n_clip > 0


def test_arm_clip_bisects_to_contact_boundary():
    # 假机械臂：x > 0.15 穿透。方块从 x=0.10 推 0.10，凡是 x 分量 > 0.05+1mm 的方向都要缩短
    sims = [FakeSim(xyz=(0.10, 0.5, 0.02), arm_x=0.15) for _ in range(40)]
    env = FakeEnv(sims)
    p = make_push(0.10)
    run_episode(p, env)
    seen = False
    for i, sim in enumerate(sims):
        r = p.episode_info(i)
        assert sim.robot_penetration() <= 0.001 + 1e-12
        if "arm" in r["push_clip"]:
            seen = True
            assert r["push_d_actual"] < 0.10
            assert r["cube_post_x"] == pytest.approx(0.151, abs=0.10 / 2 ** 12 + 1e-9)
    assert seen


def test_episode_info_has_all_perturb_columns():
    env = FakeEnv([FakeSim()])
    p = make_push(0.05)
    run_episode(p, env)
    assert set(p.episode_info(0)) == set(PERTURB_COLUMNS)
    assert set(Perturbation().episode_info(0)) <= set(PERTURB_COLUMNS)


def test_base_hooks_are_identity():
    p = Perturbation()
    obs, act = {"a": np.zeros(3)}, np.ones((2, 14))
    assert p.on_obs(0, obs, np.ones(2, bool)) is obs
    assert p.on_action(0, act, np.ones(2, bool)) is act
    assert p.on_state(0, None, np.ones(2, bool)) is None


# ----------------------------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------------------------
def test_spec_validation():
    assert normalize_spec(None) == dict(kind="none", strength=0.0, params={})
    with pytest.raises(ValueError):
        normalize_spec({"kind": "shake"})
    with pytest.raises(ValueError):
        normalize_spec({"kind": "push", "strength": -0.1})
    with pytest.raises(ValueError):
        normalize_spec({"kind": "none", "strength": 0.1})
    with pytest.raises(ValueError):
        make_perturbation({"kind": "obs_noise", "strength": 4, "params": {"bogus": 1}})
    with pytest.raises(ValueError):
        make_perturbation({"kind": "action_delay", "strength": 1.5})
    with pytest.raises(ValueError):
        make_perturbation({"kind": "push", "strength": 0.1, "params": {"t_lo": 50, "t_hi": 10}})
    with pytest.raises(ValueError):
        make_perturbation({"kind": "push", "strength": 0.1, "params": {"bogus": 1}})


def test_config_perturb_defaults_and_model_check():
    c = EvalConfig(model="act", seeds="0-9", perturb={"kind": "push", "strength": 0.05})
    assert c.perturb_active and c.perturb["params"]["t_lo"] == 20 and c.perturb["params"]["t_hi"] == 100
    c2 = EvalConfig(model="act", seeds="0-9", perturb={"kind": "push", "strength": 0.05,
                                                       "params": {"t_lo": 20, "t_hi": 100}})
    assert c.to_dict()["perturb"] == c2.to_dict()["perturb"]
    assert not EvalConfig(model="act", seeds="0-9").perturb_active
    # T8b 起 dp 的 push = PushT 推 T 形块（PushTee，见 tests/test_push_tee.py），aloha 专用参数在 dp 上报错
    with pytest.raises(ValueError):
        EvalConfig(model="dp", seeds="0-9", perturb={"kind": "push", "strength": 0.05,
                                                     "params": {"base_clearance": 0.12}})


def test_fingerprint_without_perturb_unchanged():
    from chunkre.runner import _fingerprint

    fp = _fingerprint(EvalConfig(model="act", seeds="0-9"))
    assert "perturb" not in fp           # 与 T2/T3 写出的旧 config.json 指纹相同 -> 旧结果照常续跑
    fp2 = _fingerprint(EvalConfig(model="act", seeds="0-9", perturb={"kind": "push", "strength": 0.0}))
    assert fp2["perturb"]["kind"] == "push"


def test_cli_merge(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("model: act\nseeds: 0-9\nperturb:\n  kind: push\n  strength: 0.1\n  params:\n    t_lo: 30\n",
                 encoding="utf-8")
    c = load_config(y, dict(perturb_kind=None, perturb_strength=0.02, perturb_params=parse_param_kv(["t_hi=90"])))
    assert c.perturb == dict(kind="push", strength=0.02,
                             params=dict(PushCube.DEFAULTS, t_lo=30, t_hi=90))
    c = load_config(None, dict(model="act", seeds="0-9", perturb_kind="push", perturb_strength=None,
                               perturb_params={}))
    assert c.perturb["kind"] == "push" and c.perturb["strength"] == 0.0
    assert merge_perturb(None) == {}
    with pytest.raises(ValueError):
        parse_param_kv(["t_lo"])
