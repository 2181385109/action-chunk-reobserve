"""T3：运动指标 path_length / jerk / reversal_rate、旧格式 CSV 兼容、轨迹记录器（不跑仿真）。"""
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chunkre.metrics import (EPISODE_COLUMNS, MOTION_COLUMNS, REVERSAL_EPS, jerk, motion_metrics, path_length,
                             reversal_rate, summarize_csv, summarize_rows)

ROOT = Path(__file__).resolve().parent.parent
OLD_COLUMNS = [c for c in EPISODE_COLUMNS if c not in MOTION_COLUMNS]   # T2 写出的表头


# ---------------- 三种构造的动作序列 ----------------
def still(T=50, A=14):
    """完全不动：每步动作相同。"""
    return np.tile(np.linspace(-1, 1, A), (T, 1))


def constant_velocity(T=50, A=14, v=0.01):
    """匀速：每步每个关节都 +v。"""
    return np.arange(T)[:, None] * v * np.ones((1, A))


def jitter(T=50, A=14, amp=0.01):
    """来回抖动：每步在 +amp / -amp 之间切换（每个关节）。"""
    return (np.where(np.arange(T) % 2 == 0, amp, -amp)[:, None]) * np.ones((1, A))


def test_still():
    a = still()
    assert path_length(a) == 0.0
    assert jerk(a) == 0.0
    assert math.isnan(reversal_rate(a))      # 没有任何一步在动 → 无定义（汇总时跳过）


def test_constant_velocity():
    T, A, v = 50, 14, 0.01
    a = constant_velocity(T, A, v)
    assert path_length(a) == pytest.approx((T - 1) * v * math.sqrt(A))
    assert jerk(a) == pytest.approx(0.0, abs=1e-12)
    assert reversal_rate(a) == 0.0


def test_jitter():
    T, A, amp = 50, 14, 0.01
    a = jitter(T, A, amp)
    step = 2 * amp * math.sqrt(A)                   # 每步 L2 变化
    assert path_length(a) == pytest.approx((T - 1) * step)
    assert jerk(a) == pytest.approx(2 * step)       # 二阶差分 = ±4amp 每个关节
    assert reversal_rate(a) == 1.0


def test_jitter_vs_constant_same_path_length_different_jerk_reversal():
    """同样的路径长度，抖动的 jerk / reversal 远大于匀速——这正是新增指标要区分的。"""
    T, A = 50, 14
    a = jitter(T, A, 0.01)
    b = constant_velocity(T, A, 0.02)
    assert path_length(a) == pytest.approx(path_length(b))
    assert jerk(a) > 0 and jerk(b) == pytest.approx(0.0, abs=1e-12)
    assert reversal_rate(a) == 1.0 and reversal_rate(b) == 0.0


def test_reversal_threshold_and_partial():
    # 关节 0：+,+,-,-（1 次反转 / 3 对）；关节 1：变化都小于阈值（不计入）
    d0 = np.array([0.1, 0.1, -0.1, -0.1])
    d1 = np.full(4, REVERSAL_EPS / 2) * np.array([1, -1, 1, -1])
    a = np.vstack([np.zeros(2), np.cumsum(np.stack([d0, d1], axis=1), axis=0)])
    assert reversal_rate(a) == pytest.approx(1 / 3)
    # 中间停一步再反向：静止的那步两边的对都不计入 → 无反转
    b = np.array([[0.0], [0.1], [0.1], [0.0]])
    assert math.isnan(reversal_rate(b))
    c = np.array([[0.0], [0.1], [0.2], [0.2], [0.1], [0.0]])
    assert reversal_rate(c) == 0.0


def test_short_sequences():
    assert path_length(np.zeros((1, 3))) == 0.0
    assert math.isnan(jerk(np.zeros((2, 3))))
    assert math.isnan(reversal_rate(np.zeros((2, 3))))
    m = motion_metrics([0.0, 1.0, 3.0])          # 一维也行
    assert m["path_length"] == 3.0 and m["jerk"] == 1.0 and m["reversal_rate"] == 0.0


# ---------------- 汇总与旧格式 CSV ----------------
def _rows(n=20, B=10, with_motion=True):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        r = dict(batch_idx=i // B, env_idx=i % B, seed=i, model="act", mode="fixed", k=1,
                 success=int(rng.random() < 0.5), steps=400, max_steps=400, capped=0,
                 sum_reward=1.0, max_reward=1.0, n_policy_calls=400, smoothness=0.01,
                 wall_s=1.0, batch_wall_s=10.0, batch_policy_s=1.0, batch_env_s=5.0)
        if with_motion:
            r.update(path_length=float(rng.random()), jerk=float(rng.random()),
                     reversal_rate=float("nan") if i == 0 else float(rng.random()))
        rows.append(r)
    return rows


def _write(p, rows, cols):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_summary_motion_means(tmp_path):
    rows = _rows()
    _write(tmp_path / "episodes.csv", rows, EPISODE_COLUMNS)
    m = summarize_csv(tmp_path / "episodes.csv")
    assert m["mean_path_length"] == pytest.approx(np.mean([r["path_length"] for r in rows]))
    assert m["mean_reversal_rate"] == pytest.approx(np.nanmean([r["reversal_rate"] for r in rows]))  # nan 跳过
    direct = summarize_rows(rows)
    assert m.keys() == direct.keys()
    assert all(v == direct[k] or (v != v and direct[k] != direct[k]) for k, v in m.items())


def test_old_format_csv_summarizes_without_error(tmp_path):
    """T2 的旧 CSV（无运动指标列）：汇总不报错、键与旧 summary 完全相同，summarize.py 退出码 0。"""
    rows = _rows(with_motion=False)
    _write(tmp_path / "episodes.csv", rows, OLD_COLUMNS)
    m = summarize_csv(tmp_path / "episodes.csv")
    assert not any(k.startswith("mean_" + c) for c in MOTION_COLUMNS for k in m)
    (tmp_path / "summary.json").write_text(json.dumps(dict(metrics=m)), encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize.py"), str(tmp_path)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "旧格式" in r.stdout


def test_real_t2_dirs_still_summarize():
    """仓库里 T2 的真实结果目录（旧格式）仍能被 summarize.py 校验为一致。"""
    dirs = sorted(d for d in (ROOT / "results" / "t2_checks").glob("act_*")
                  if (d / "summary.json").exists() and (d / "episodes.csv").exists())
    if not dirs:
        pytest.skip("没有 T2 结果目录")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize.py"), *map(str, dirs)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr


def test_resume_old_format_keeps_header(tmp_path):
    """续跑旧格式 CSV：删掉不完整批次时保持旧表头；run_eval 沿用旧表头（运动指标列不写进去）。"""
    from chunkre.config import EvalConfig
    from chunkre.metrics import csv_columns
    from chunkre.runner import _read_done_batches

    cfg = EvalConfig(model="act", seeds="0-29", batch_size=10)
    rows = _rows(n=25, with_motion=False)
    p = tmp_path / "episodes.csv"
    _write(p, rows, OLD_COLUMNS)
    assert _read_done_batches(p, cfg) == {0, 1}
    assert csv_columns(p) == OLD_COLUMNS
    # 旧表头 + extrasaction="ignore" 追加新行：新行多出的运动指标被丢掉，文件仍是合法旧格式
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_columns(p), extrasaction="ignore")
        w.writerow(dict(_rows(n=1)[0], batch_idx=2))
    with open(p, newline="", encoding="utf-8") as f:
        back = list(csv.DictReader(f))
    assert len(back) == 21 and all(None not in r for r in back)


def test_logging_fields_not_in_fingerprint():
    """--log-traj / --save-frames 不进续跑指纹：T2 的旧 config.json 照样能续跑。"""
    from chunkre.config import EvalConfig
    from chunkre.runner import _fingerprint

    a = EvalConfig(model="act", seeds="0-9", batch_size=10, k=1)
    b = EvalConfig(model="act", seeds="0-9", batch_size=10, k=1, log_traj=True, save_frames="0,3", frame_every=5)
    assert _fingerprint(a) == _fingerprint(b)
    assert b.save_frames == [0, 3]
    old = json.loads((ROOT / "results" / "t2_checks" / "act_v4_k1" / "config.json").read_text(encoding="utf-8"))
    assert old["eval_config_fingerprint"] == _fingerprint(a)


# ---------------- 轨迹记录器（假环境） ----------------
def test_recorder_chunk_idx_and_npz(tmp_path):
    from chunkre.traj import TrajRecorder

    B, A, H, k = 2, 3, 10, 4
    rec = TrajRecorder(tmp_path, 0, [5, 6], log_traj=True)
    env = SimpleNamespace(envs=[SimpleNamespace(unwrapped=SimpleNamespace())] * B)   # 不是 aloha：无 cube_pos
    T = 10
    for t in range(T):
        alive = np.array([True, t < 6])
        rep = t % k == 0
        chunk = np.arange(B * H * A, dtype=np.float32).reshape(B, H, A) + t if rep else None
        obs = dict(agent_pos=np.full((B, A), t, dtype=np.float32))
        rec.record(t, obs, env, np.full((B, A), t, dtype=np.float32), alive, rep, chunk)
    rec.finish(np.array([10, 6]))
    z = np.load(tmp_path / "traj" / "batch_0000.npz")
    assert list(z["chunk_idx"]) == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    assert list(z["repredict"]) == [t % k == 0 for t in range(T)]
    assert z["action"].shape == (T, B, A) and np.isnan(z["action"][6:, 1]).all()
    assert list(z["chunk_offsets"]) == [0, 1, 4, 9]          # 超出块长 10 的位置丢掉
    assert z["chunk_sel"].shape == (T, B, 4, A)
    assert np.isnan(z["chunk_sel"][1]).all() and not np.isnan(z["chunk_sel"][4, 0]).any()
    assert "cube_pos" not in z.files


def test_frames_and_gif_size(tmp_path):
    from chunkre.traj import TrajRecorder

    rng = np.random.default_rng(0)
    rec = TrajRecorder(tmp_path, 0, [3, 4], log_traj=False, frame_seeds=[4], frame_every=10)
    assert rec.active
    for t in range(400):
        img = rng.integers(0, 255, size=(2, 480, 640, 3), dtype=np.uint8)   # 噪声图：最难压缩
        rec.record(t, dict(pixels=dict(top=img)), None, None, np.array([True, True]), False, None)
    rec.finish(np.array([400, 400]))
    pngs = sorted((tmp_path / "frames" / "seed4").glob("*.png"))
    assert len(pngs) == 40 and not (tmp_path / "frames" / "seed3").exists()
    assert (tmp_path / "frames" / "seed4.gif").stat().st_size <= 2 * 1024 * 1024
