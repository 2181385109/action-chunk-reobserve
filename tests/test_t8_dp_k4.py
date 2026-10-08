"""T8c K2 “T 块附近方向切换次数”的手算测试（不跑仿真）。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEE_LOCAL = [np.array([(-60, 30), (60, 30), (60, 0), (-60, 0)], float),
             np.array([(-15, 30), (-15, 120), (15, 120), (15, 30)], float)]


def _mod():
    spec = importlib.util.spec_from_file_location("t8_dp_k4", ROOT / "scripts" / "t8_dp_k4.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["t8_dp_k4"] = m
    spec.loader.exec_module(m)
    return m


def _run(path, repredict=None, block=(256.0, 256.0, 0.0)):
    m = _mod()
    pos = np.array(path, float)
    n = len(pos)
    blk = np.tile(np.array(block), (n, 1))
    rep = np.zeros(n, bool) if repredict is None else np.asarray(repredict, bool)
    return m.near_t_switches(pos, blk, rep, n, TEE_LOCAL)


def test_back_and_forth_near_t_counts_every_reversal():
    # T 在 (256,256)，智能体在横条下方 y=230（间隙 = 256-230-15 = 11 <= 30）左右来回：+10, +10, -10, -10, +10
    xs = [200, 210, 220, 210, 200, 210]
    r = _run([(x, 230.0) for x in xs])
    assert r["n_flips"] == 2 and r["n_moving_steps"] == 5 and r["n_near_moving_steps"] == 5


def test_far_from_t_or_standing_still_does_not_count():
    xs = [200, 210, 220, 210, 200, 210]
    far = _run([(x, 100.0) for x in xs])                    # 间隙 141 > 30
    assert far["n_flips"] == 0 and far["n_near_moving_steps"] == 0
    still = _run([(200.0, 230.0)] * 3 + [(210.0, 230.0)] + [(200.0, 230.0)])   # 静止步不算，但前后两个在动的步相反 -> 1 次
    assert still["n_flips"] == 1
    sub = _run([(200.0, 230.0), (200.4, 230.0), (200.0, 230.0), (200.4, 230.0)])   # 每步 < 1 单位：都不算在动
    assert sub["n_moving_steps"] == 0 and sub["n_flips"] == 0


def test_perpendicular_turn_is_not_a_flip_and_boundary_split():
    turn = _run([(200, 230), (210, 230), (210, 240)])      # 右 -> 上，夹角 90°，cos = 0，不算
    assert turn["n_flips"] == 0
    # 步 0: +10，步 1: -10（t=1 是重新预测的那一步 -> 块边界），步 2: +10（块内部）
    r = _run([(200, 230), (210, 230), (200, 230), (210, 230)], repredict=[0, 1, 0, 0])
    assert r["n_flips"] == 2 and r["n_flips_boundary"] == 1 and r["n_flips_interior"] == 1
