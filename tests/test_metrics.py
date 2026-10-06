import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from chunkre.metrics import EPISODE_COLUMNS, action_smoothness, summarize_csv, summarize_rows, wilson_ci

ROOT = Path(__file__).resolve().parent.parent


def test_smoothness_definition():
    a = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0], [0.0, 0.0]])
    # 相邻差的 L2 范数：5, 0, 5 → 平均 10/3
    assert action_smoothness(a) == pytest.approx(10 / 3)
    assert action_smoothness(np.zeros((5, 14))) == 0.0
    assert math.isnan(action_smoothness(np.zeros((1, 3))))
    assert action_smoothness([1.0, 2.0, 4.0]) == pytest.approx(1.5)  # 一维动作


def test_smoothness_matches_runner_incremental():
    """runner.run_batch 里逐步累加的算法与 action_smoothness 一致（含提前结束的环境）。"""
    rng = np.random.default_rng(0)
    acts = rng.normal(size=(50, 3, 2))           # (步, 环境, 维度)
    steps = np.array([50, 20, 1])                # 各环境存活步数
    diff_sum = np.zeros(3)
    diff_n = np.zeros(3, dtype=int)
    prev = None
    for t in range(50):
        alive = t < steps
        if prev is not None:
            d = np.linalg.norm(acts[t] - prev, axis=1)
            diff_sum[alive] += d[alive]
            diff_n[alive] += 1
        prev = acts[t]
    inc = np.where(diff_n > 0, diff_sum / np.maximum(diff_n, 1), np.nan)
    for i in range(3):
        ref = action_smoothness(acts[: steps[i], i])
        assert (math.isnan(ref) and math.isnan(inc[i])) or inc[i] == pytest.approx(ref)


def test_wilson():
    lo, hi = wilson_ci(83, 100)
    assert lo == pytest.approx(0.7444, abs=1e-3) and hi == pytest.approx(0.8909, abs=1e-3)
    assert wilson_ci(0, 10)[0] == pytest.approx(0.0) and wilson_ci(10, 10)[1] == pytest.approx(1.0)
    lo, hi = wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert all(math.isnan(x) for x in wilson_ci(0, 0))


def _fake_rows(n=20, B=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        b = i // B
        rows.append(dict(
            batch_idx=b, env_idx=i % B, seed=i, model="act", mode="fixed", k=10,
            success=int(rng.random() < 0.7), steps=int(rng.integers(100, 401)), max_steps=400, capped=0,
            sum_reward=float(rng.random() * 100), max_reward=float(rng.integers(0, 5)),
            n_policy_calls=int(rng.integers(10, 41)), smoothness=float(rng.random() / 3),
            wall_s=float(rng.random() * 50), batch_wall_s=40.0 + b, batch_policy_s=3.0, batch_env_s=30.0,
        ))
    return rows


def test_summary_recomputable_from_csv(tmp_path):
    rows = _fake_rows()
    p = tmp_path / "episodes.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EPISODE_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    direct = summarize_rows(rows)            # 原始数值
    from_csv = summarize_csv(p)              # 从 CSV 字符串读回
    assert direct == from_csv
    assert from_csv["n_episodes"] == 20
    assert from_csv["n_success"] == sum(r["success"] for r in rows)
    assert from_csv["n_batches"] == 2 and from_csv["total_batch_wall_s"] == 81.0
    lo, hi = from_csv["success_ci95_wilson"]
    assert lo <= from_csv["success_rate"] <= hi
    # summary.json 写出再由 scripts/summarize.py 校验：一致时退出码 0，被改动时 1
    (tmp_path / "summary.json").write_text(json.dumps(dict(metrics=from_csv)), encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize.py"), str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    bad = dict(from_csv, n_success=from_csv["n_success"] + 1)
    (tmp_path / "summary.json").write_text(json.dumps(dict(metrics=bad)), encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize.py"), str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 1


def test_resume_drops_incomplete_batch(tmp_path):
    from chunkre.config import EvalConfig
    from chunkre.runner import _read_done_batches

    cfg = EvalConfig(model="act", seeds="0-29", batch_size=10)
    rows = _fake_rows(n=25)                  # 批 0、1 完整，批 2 只有 5 行（中断）
    p = tmp_path / "episodes.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EPISODE_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    assert _read_done_batches(p, cfg) == {0, 1}
    with open(p, newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 20
