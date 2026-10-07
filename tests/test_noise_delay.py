"""T5 观测噪声 / 动作延迟单元测试（不跑仿真：观测和动作用假的数组）。"""
import random

import numpy as np
import pytest

import chunkre.perturb as P
from chunkre.config import EvalConfig
from chunkre.perturb import ActionDelay, ObsNoise, episode_rng, make_perturbation

H, W = 12, 16


def aloha_obs(B, value=128, t=0):
    """aloha 风格：pixels 是 {相机: (B,H,W,3) uint8}，agent_pos 14 维。"""
    img = np.full((B, H, W, 3), value, dtype=np.uint8)
    return {"pixels": {"top": img}, "agent_pos": np.arange(B * 14, dtype=np.float64).reshape(B, 14) + 100 * t}


def pusht_obs(B, value=128, t=0):
    """PushT 风格：pixels 直接是 (B,H,W,3) uint8，agent_pos 2 维。"""
    return {"pixels": np.full((B, H, W, 3), value, dtype=np.uint8),
            "agent_pos": np.arange(B * 2, dtype=np.float64).reshape(B, 2) + 100 * t}


def _global_states():
    import torch

    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _same(a, b):
    import torch

    assert a[0] == b[0]
    assert all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(a[1], b[1]))
    assert torch.equal(a[2], b[2])


# ----------------------------------------------------------------------------------------------
# 强度 0 旁路
# ----------------------------------------------------------------------------------------------
def test_noise_zero_is_full_bypass(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("σ=0 不应建随机数")

    monkeypatch.setattr(P, "episode_rng", boom)
    p = make_perturbation({"kind": "obs_noise", "strength": 0})
    p.begin_batch(None, [0, 1, 2])
    alive = np.ones(3, bool)
    for t in range(5):
        obs = aloha_obs(3)
        assert p.on_obs(t, obs, alive) is obs
        a = np.ones((3, 14), np.float32)
        assert p.on_action(t, a, alive) is a
    assert p.rngs == []


def test_delay_zero_is_full_bypass():
    p = make_perturbation({"kind": "action_delay", "strength": 0})
    assert not p.touches_action
    p.begin_batch(None, [0, 1])
    alive = np.ones(2, bool)
    for t in range(5):
        obs = aloha_obs(2, t=t)
        assert p.on_obs(t, obs, alive) is obs
        a = np.full((2, 14), t, np.float32)
        assert p.on_action(t, a, alive) is a
    assert p.hold is None and len(p.buf) == 0


# ----------------------------------------------------------------------------------------------
# 观测噪声
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("make_obs", [aloha_obs, pusht_obs])
def test_noise_dtype_clip_and_agent_pos_untouched(make_obs):
    p = ObsNoise(300.0)            # 很大的 σ：大部分像素被截断到 0 / 255
    p.begin_batch(None, [0, 1])
    obs = make_obs(2)
    orig = {k: (v.copy() if isinstance(v, np.ndarray) else {c: x.copy() for c, x in v.items()}) for k, v in obs.items()}
    new = p.on_obs(0, obs, np.ones(2, bool))
    img = new["pixels"]["top"] if isinstance(new["pixels"], dict) else new["pixels"]
    assert img.dtype == np.uint8 and img.shape == (2, H, W, 3)
    assert set(np.unique(img)) <= set(range(256)) and (img == 0).any() and (img == 255).any()
    assert new["agent_pos"] is obs["agent_pos"]
    np.testing.assert_array_equal(new["agent_pos"], orig["agent_pos"])
    # 不原地改环境给的数组
    src = obs["pixels"]["top"] if isinstance(obs["pixels"], dict) else obs["pixels"]
    ref = orig["pixels"]["top"] if isinstance(orig["pixels"], dict) else orig["pixels"]
    np.testing.assert_array_equal(src, ref)
    assert type(new["pixels"]) is type(obs["pixels"])


def test_noise_empirical_std_close_to_sigma():
    for sigma in (2.0, 8.0, 20.0):
        p = ObsNoise(sigma)
        p.begin_batch(None, [5])
        obs = {"pixels": np.full((1, 64, 64, 3), 128, np.uint8), "agent_pos": np.zeros((1, 2))}
        d = p.on_obs(0, obs, np.ones(1, bool))["pixels"].astype(np.float64) - 128.0
        # 中间灰不会被截断：std ≈ sqrt(σ² + 1/12)（取整），12288 个样本，相对误差 < 3%
        assert abs(d.std() - np.sqrt(sigma ** 2 + 1 / 12)) / sigma < 0.03
        assert abs(d.mean()) < 0.1 * sigma


def test_noise_independent_per_step_and_per_episode():
    p = ObsNoise(10.0)
    p.begin_batch(None, [3, 3, 4])            # 两个相同种子 + 一个不同种子
    alive = np.ones(3, bool)
    a = p.on_obs(0, pusht_obs(3), alive)["pixels"].astype(int)
    b = p.on_obs(1, pusht_obs(3), alive)["pixels"].astype(int)
    assert np.array_equal(a[0], a[1])          # 同一种子同一步：同样的噪声
    assert not np.array_equal(a[0], a[2])      # 不同种子：不同
    assert not np.array_equal(a[0], b[0])      # 同一回合不同步：不同


def test_noise_same_seed_same_noise_regardless_of_batch_position_and_dead_envs():
    p1, p2 = ObsNoise(6.0), ObsNoise(6.0)
    p1.begin_batch(None, [7, 8])
    p2.begin_batch(None, [9, 7])
    seq1, seq2 = [], []
    for t in range(4):
        alive2 = np.array([t < 2, True])       # p2 里第 0 个回合提前结束
        o1 = p1.on_obs(t, aloha_obs(2), np.ones(2, bool))["pixels"]["top"]
        o2 = p2.on_obs(t, aloha_obs(2), alive2)["pixels"]["top"]
        seq1.append(o1[0].copy())
        seq2.append(o2[1].copy())
        if t >= 2:                              # 已结束的回合不加噪声
            np.testing.assert_array_equal(o2[0], aloha_obs(2)["pixels"]["top"][0])
    for x, y in zip(seq1, seq2):
        np.testing.assert_array_equal(x, y)


def test_noise_uses_episode_rng_stream():
    """噪声 = episode_rng(种子, 'obs_noise') 按步顺序抽的 float32 标准正态（中间灰、σ 小时无截断可逐元素对照）。"""
    sigma = 3.0
    p = ObsNoise(sigma)
    p.begin_batch(None, [11])
    rng = episode_rng(11, "obs_noise")
    for t in range(3):
        out = p.on_obs(t, pusht_obs(1), np.ones(1, bool))["pixels"][0]
        n = rng.standard_normal((H, W, 3), dtype=np.float32)
        exp = np.clip(np.rint(np.float32(128) + np.float32(sigma) * n), 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(out, exp)


def test_noise_requires_pixels():
    p = ObsNoise(1.0)
    p.begin_batch(None, [0])
    with pytest.raises(ValueError):
        p.on_obs(0, {"agent_pos": np.zeros((1, 2))}, np.ones(1, bool))


# ----------------------------------------------------------------------------------------------
# 动作延迟
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 3, 7])
@pytest.mark.parametrize("make_obs,A", [(aloha_obs, 14), (pusht_obs, 2)])
def test_delay_fifo_order_and_hold(d, make_obs, A):
    B, T = 3, 20
    p = make_perturbation({"kind": "action_delay", "strength": d})
    assert p.touches_action and p.d == d
    p.begin_batch(None, list(range(B)))
    alive = np.ones(B, bool)
    hold = make_obs(B, t=0)["agent_pos"]
    policy, executed = [], []
    for t in range(T):
        p.on_obs(t, make_obs(B, t=t), alive)        # 之后的 agent_pos 变了，保持位置仍用第 0 步的
        a = np.random.default_rng(t).normal(size=(B, A)).astype(np.float32)
        policy.append(a.copy())
        out = p.on_action(t, a, alive)
        a[:] = -999                                  # 调用后改策略的数组，不应影响缓冲
        assert out.dtype == np.float32 and out.shape == (B, A)
        executed.append(out.copy())
    for t in range(T):
        if t < d:
            np.testing.assert_array_equal(executed[t], hold.astype(np.float32))
        else:
            np.testing.assert_array_equal(executed[t], policy[t - d])
    assert len(p.buf) == d


def test_delay_resets_between_batches():
    p = ActionDelay(2)
    alive = np.ones(1, bool)
    for batch in range(2):
        p.begin_batch(None, [batch])
        obs = pusht_obs(1, t=batch)
        p.on_obs(0, obs, alive)
        out = p.on_action(0, np.full((1, 2), 5.0, np.float32), alive)
        np.testing.assert_array_equal(out, obs["agent_pos"].astype(np.float32))   # 新批又从保持位置开始


def test_delay_shape_mismatch_and_bad_strength():
    p = ActionDelay(2)
    p.begin_batch(None, [0])
    p.on_obs(0, pusht_obs(1), np.ones(1, bool))
    with pytest.raises(ValueError):
        p.on_action(0, np.zeros((1, 14), np.float32), np.ones(1, bool))
    with pytest.raises(ValueError):
        ActionDelay(2.5)
    with pytest.raises(ValueError):
        ActionDelay(2, {"x": 1})


# ----------------------------------------------------------------------------------------------
# 随机数不碰全局状态
# ----------------------------------------------------------------------------------------------
def test_noise_and_delay_do_not_touch_global_rng():
    import torch

    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    before = _global_states()
    for spec in ({"kind": "obs_noise", "strength": 12}, {"kind": "action_delay", "strength": 4}):
        p = make_perturbation(spec)
        p.begin_batch(None, [0, 1, 2])
        alive = np.ones(3, bool)
        for t in range(10):
            p.on_obs(t, aloha_obs(3, t=t), alive)
            p.on_action(t, np.zeros((3, 14), np.float32), alive)
        p.end_batch(np.full(3, 10))
    _same(before, _global_states())


# ----------------------------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["act", "dp"])
def test_config_accepts_both_models(model):
    c = EvalConfig(model=model, seeds="0-9", perturb={"kind": "obs_noise", "strength": 8})
    assert c.perturb == dict(kind="obs_noise", strength=8.0, params={})
    c = EvalConfig(model=model, seeds="0-9", perturb={"kind": "action_delay", "strength": 2})
    assert c.perturb == dict(kind="action_delay", strength=2.0, params={})
    assert c.perturb_active


def test_describe_and_episode_info():
    for p in (ObsNoise(4), ActionDelay(3)):
        p.begin_batch(None, [0])
        assert p.describe()["kind"] == p.kind
        info = p.episode_info(0)
        assert info["perturb_kind"] == p.kind and set(info) <= set(P.PERTURB_COLUMNS)
