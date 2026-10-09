"""T13 失败案例分析的纯计算（src/chunkre/failure.py、scripts/failure_tools.py、analyze_failure 的挑选 / 拼图）。不跑仿真。"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from chunkre import failure as F
from chunkre import traj

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_failure as AF  # noqa: E402
import failure_tools as FT  # noqa: E402


def cube_z(lift=0.0, n=100, at=50):
    z = np.full(n, 0.02)
    z[:5] = np.linspace(0.05, 0.016, 5)     # 开局下落
    z[at] = 0.02 + lift
    return z


def test_rest_height_ignores_spawn_drop():
    assert F.rest_height(cube_z()) == pytest.approx(0.02)
    assert F.max_lift(cube_z()) == pytest.approx(0.0)      # 第 0 步 z=0.05 不算抬起


def test_classify_thresholds():
    assert F.H_LIFT == 0.04
    assert F.classify_episode(True, cube_z(0.0)) == "success"
    assert F.classify_episode(False, cube_z(0.0191)) == "never_lifted"
    assert F.classify_episode(False, cube_z(0.0401)) == "lifted_failed"
    assert F.classify_episode(False, cube_z(0.04)) == "never_lifted"   # 严格大于


def test_first_repredict_after_excludes_push_step():
    rp = np.zeros(200, dtype=bool)
    rp[[0, 20, 40, 60]] = True
    assert F.first_repredict_after(rp, 40, 200) == 20     # 第 40 步那次用的是推之前的画面
    assert F.first_repredict_after(rp, 39, 200) == 1
    assert F.first_repredict_after(rp, 60, 200) is None
    assert F.first_repredict_after(rp, 10, 15) is None    # 回合在下一次重新预测前结束


def test_close_events_need_opening_first():
    g = np.array([0.1, 0.1, 0.4, 0.9, 1.1, 0.7, 0.45, 0.2, 0.3, 0.6, 0.2, 0.85, 0.49, np.nan, 0.1])
    assert F.close_events(g) == [6, 12]            # 开局 0.1 不算；0.6 没到 0.8 不重新武装；nan 之后不看
    assert F.first_close_after(g, 6) == 12
    assert F.first_close_after(g, 5) == 6
    assert F.first_close_after(g, 12) is None


def test_grip_point_and_horiz_dist():
    tips = np.zeros((1, 4, 3))
    tips[0, 2] = [0.1, 0.5, 0.3]
    tips[0, 3] = [0.1, 0.52, 0.1]
    gp = F.grip_point(tips)
    assert np.allclose(gp[0], [0.1, 0.51, 0.2])
    assert F.horiz_dist(gp[0], [0.13, 0.55, 0.02]) == pytest.approx(0.05)   # 只看 xy


def test_switch_jumps_no_delay():
    a = np.zeros((60, 14))
    a[:, 0] = np.arange(60) * 0.01
    a[20:, 0] += 0.5          # 第 20 步新块：关节跳 0.51
    a[40:, 13] += 0.3         # 第 40 步新块：夹爪跳 0.3
    rp = np.zeros(60, dtype=bool)
    rp[[0, 20, 40]] = True
    j = F.switch_jumps(a, rp, 60, delay=0)
    assert list(j["t_exec"]) == [20, 40]
    assert j["arm"][0] == pytest.approx(0.51)
    assert j["grip"][1] == pytest.approx(0.3)
    assert j["full"][1] == pytest.approx(math.hypot(0.01, 0.3))
    assert len(j["base_arm"]) == 59 - 2 and np.allclose(j["base_arm"], 0.01)


def test_switch_jumps_delay_offsets_and_hold():
    d = 10
    a = np.zeros((70, 14))
    a[d:, 1] = 0.2 + np.arange(70 - d) * 0.01     # 保持位置 0 -> 第 d 步开始执行策略输出
    a[d + 20:, 1] -= 0.3                           # 第 20 步预测的块在第 30 步才执行
    rp = np.zeros(70, dtype=bool)
    rp[[0, 20, 40, 60]] = True
    j = F.switch_jumps(a, rp, 70, delay=d)
    assert list(j["t_exec"]) == [30, 50]           # 60 + 10 = 70 超出回合
    assert j["arm"][0] == pytest.approx(0.29)
    assert np.allclose(j["base_arm"], 0.01)        # 保持阶段和保持 -> 第一个策略动作的 0.2 跳不算


def test_backtrack_cos():
    a = np.zeros((30, 14))
    a[:, 0] = np.arange(30) * 0.01
    a[20:, 0] -= 0.2                               # 往回跳
    a2 = a.copy()
    a2[20:, 0] += 0.4                              # 往前跳
    assert F.jump_backtrack_cos(a, [20])[0] == pytest.approx(-1.0)
    assert F.jump_backtrack_cos(a2, [20])[0] == pytest.approx(1.0)
    assert np.isnan(F.jump_backtrack_cos(a, [3])[0])


def test_describe():
    d = F.describe([1, 2, 3, 4, None, float("nan")])
    assert d["n"] == 4 and d["median"] == 2.5 and d["min"] == 1 and d["max"] == 4
    assert F.describe([])["n"] == 0


def test_quat_to_mat():
    assert np.allclose(traj._quat_to_mat(np.array([1.0, 0, 0, 0])), np.eye(3))
    c = math.cos(math.pi / 4)
    R = traj._quat_to_mat(np.array([c, 0, 0, c]))     # 绕 z 转 90°
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])


def test_failure_tools_args():
    seeds = list(range(7000, 7050))
    a = FT.cell_args("k20__push0.06", seeds)
    assert a[:2] == ["--k", "20"] and "push" in a and "t_hi=100" in a
    assert a[a.index("--save-frames") + 1] == ",".join(str(s) for s in range(7000, 7010))
    b = FT.cell_args("k50__delay10", seeds)
    assert b[:2] == ["--k", "50"] and b[b.index("--perturb-strength") + 1] == "10"
    assert b[b.index("--save-frames") + 1] == "7000,7001,7002"
    assert b[b.index("--frame-every") + 1] == "3"
    c = FT.cell_args("k100__none", seeds)
    assert "--perturb" not in c
    assert FT.CELLS == AF.CELLS


def test_pick_pairs_prefers_frames():
    def ep(fr):
        return dict(frames=fr)
    cells = {"k20__push0.06": dict(eps={1: ep([]), 2: ep([0, 3]), 3: ep([0]), 4: ep([0])}),
             "k100__push0.06": dict(eps={1: ep([0]), 2: ep([0]), 3: ep([]), 4: ep([0])})}
    assert AF.pick_pairs(cells, [1, 2, 3, 4]) == [2, 4]
    assert AF.pick_pairs(cells, [1, 3]) == [1, 3]


def test_nearest_frame():
    fr = [0, 3, 6, 9]
    assert AF._nearest_frame(fr, 5, "le") == 3
    assert AF._nearest_frame(fr, 5, "ge") == 6
    assert AF._nearest_frame(fr, 7, "near") == 6
    assert AF._nearest_frame([], 7, "near") is None


def test_montage(tmp_path):
    from PIL import Image

    p = tmp_path / "a.png"
    Image.new("RGB", (640, 480), "red").save(p)
    out = tmp_path / "m.png"
    AF.montage([[(p, "x"), (None, "missing")], [(p, "y")]], out, width=100)
    im = Image.open(out)
    assert im.size == (200, 2 * (75 + 18))
