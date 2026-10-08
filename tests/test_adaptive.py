"""T11a 自适应重新观察规则：差异计算、min_gap、块用完、调用次数计数、配置 / 指纹 / 汇总。不跑仿真。"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chunkre.adaptive import (ADAPTIVE_COLUMNS, CAUSE_DIFF, CAUSE_EXHAUST, CAUSE_INITIAL, AdaptiveController,
                              load_checks, normalize_params, parse_tau, plan_diff)
from chunkre.config import EvalConfig, load_config, validate_k
from chunkre.metrics import summarize_rows

ROOT = Path(__file__).resolve().parent.parent
L, A = 100, 14
ACT = SimpleNamespace(chunk_size=100, n_action_steps=100)


def fake_chunks(t, B, offset=0.0):
    """第 t 步预测的块：chunk[i, s, :] = t*1000 + s + 环境 i*1e6 + offset（能从动作值反推是哪次预测的第几个动作）。"""
    s = np.arange(L, dtype=np.float64)[None, :, None]
    i = np.arange(B, dtype=np.float64)[:, None, None]
    return np.broadcast_to(t * 1000.0 + s + i * 1e6 + offset, (B, L, A)).astype(np.float32)


def run(params, B=2, T=400, alive_fn=None, chunk_fn=fake_chunks):
    ctrl = AdaptiveController(B, params, L)
    acts, calls = [], np.zeros(B, dtype=np.int64)
    for t in range(T):
        alive = np.ones(B, dtype=bool) if alive_fn is None else alive_fn(t)
        need = ctrl.need_predict(alive)
        ch = chunk_fn(t, B) if need.any() else None
        acts.append(ctrl.step(t, alive, ch))
        calls[alive] += ctrl.called[alive]
    return ctrl, np.stack(acts), calls


# ---------------------------------------------------------------- 差异计算
def test_plan_diff_mean_l2_over_h_cmp():
    old = np.zeros((L, 2))
    cand = np.zeros((L, 2))
    cand[:, 0] = 3.0
    cand[:, 1] = 4.0          # 每步 L2 = 5
    d, ov = plan_diff(old, 0, cand, 20)
    assert ov == 20 and d == pytest.approx(5.0)
    # 只看旧计划下标 j..j+h-1 与候选 0..h-1
    old2 = np.arange(L, dtype=np.float64)[:, None] * np.ones((1, 2))
    cand2 = np.full((L, 2), 30.0)
    d2, ov2 = plan_diff(old2, 30, cand2, 3)     # 旧 30,31,32 vs 候选 30 → 0,1,2 每维；L2 = 0, √2, 2√2
    assert ov2 == 3 and d2 == pytest.approx((0 + math.sqrt(2) + 2 * math.sqrt(2)) / 3)


def test_plan_diff_overlap_when_old_plan_short():
    old = np.zeros((L, 1))
    cand = np.ones((L, 1))
    cand[:5] = 2.0
    d, ov = plan_diff(old, 95, cand, 20)          # 旧计划只剩 5 步
    assert ov == 5 and d == pytest.approx(2.0)
    d0, ov0 = plan_diff(old, 100, cand, 20)
    assert ov0 == 0 and math.isnan(d0)


def test_plan_diff_is_float64_and_symmetric():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(L, A)).astype(np.float32), rng.normal(size=(L, A)).astype(np.float32)
    d1, _ = plan_diff(a, 0, b, 20)
    d2, _ = plan_diff(b, 0, a, 20)
    ref = np.linalg.norm(b[:20].astype(np.float64) - a[:20].astype(np.float64), axis=1).mean()
    assert d1 == d2 == pytest.approx(ref, rel=0, abs=0)


# ---------------------------------------------------------------- 参数
def test_params_defaults_and_tau_inf():
    p = normalize_params(dict(m=10, tau="inf"))
    assert p == dict(m=10, tau=math.inf, h_cmp=20, min_gap=20)
    assert parse_tau(".inf") == math.inf and parse_tau("Infinity") == math.inf and parse_tau(0) == 0.0
    assert normalize_params(dict(m=5, tau=0.1, min_gap=5, h_cmp=10)) == dict(m=5, tau=0.1, h_cmp=10, min_gap=5)


@pytest.mark.parametrize("bad", [dict(tau=1), dict(m=10), dict(m=0, tau=1), dict(m=10, tau=-1),
                                 dict(m=10, tau=1, min_gap=-1), dict(m=10, tau=1, h_cmp=0), dict(m=10, tau=1, x=1)])
def test_params_invalid(bad):
    with pytest.raises(ValueError):
        normalize_params(bad)


# ---------------------------------------------------------------- tau = 无穷大：与 k=100 完全相同
def test_tau_inf_actions_equal_k100_and_calls():
    ctrl, acts, calls = run(dict(m=10, tau="inf"))
    t = np.arange(400)
    ref = (t // 100 * 100) * 1000.0 + t % 100          # k=100：第 0/100/200/300 步预测，执行第 t%100 个
    assert np.array_equal(acts[:, 0, 0], ref.astype(np.float32))
    # 比较在 j=10..90（每块 9 次），换计划只有块用完（100、200、300）
    assert ctrl.n_checks.tolist() == [36, 36]
    assert ctrl.n_exhaust.tolist() == [3, 3] and ctrl.n_diff.tolist() == [0, 0]
    assert calls.tolist() == ctrl.n_calls.tolist() == [1 + 3 + 36] * 2
    rp = [(e, t, c) for e, t, c in ctrl.replans if e == 0]
    assert rp == [(0, 0, CAUSE_INITIAL), (0, 100, CAUSE_EXHAUST), (0, 200, CAUSE_EXHAUST), (0, 300, CAUSE_EXHAUST)]
    assert all(not sw for *_, sw in ctrl.checks)
    js = sorted({c[2] for c in ctrl.checks})
    assert js == list(range(10, 100, 10))


# ---------------------------------------------------------------- tau = 0, min_gap = m：间隔恰好 m
@pytest.mark.parametrize("m", [10, 20, 30])
def test_tau0_interval_exactly_m(m):
    ctrl, acts, calls = run(dict(m=m, tau=0, min_gap=m), B=1)
    t_rp = [t for e, t, c in ctrl.replans]
    assert np.all(np.diff(t_rp) == m)
    assert all(c == CAUSE_DIFF for e, t, c in ctrl.replans[1:])
    # 换计划那一步执行的是候选块的第 1 个动作（下标 0，值 = t*1000）
    for t in t_rp:
        assert acts[t, 0, 0] == np.float32(t * 1000.0)
    # 每次调用都换计划：调用次数 = 1 + 换计划次数，没有“块用完”
    assert ctrl.n_exhaust[0] == 0 and calls[0] == 1 + ctrl.n_diff[0] == 1 + ctrl.n_checks[0]


# ---------------------------------------------------------------- min_gap
def test_min_gap_blocks_early_switch():
    ctrl, acts, _ = run(dict(m=10, tau=0, min_gap=20), B=1, T=200)
    t_rp = [t for e, t, c in ctrl.replans]
    assert t_rp == list(range(0, 200, 20))                # j=10 的比较不能换（j < min_gap），j=20 换
    ineligible = [c for c in ctrl.checks if not c[5]]
    assert ineligible and all(c[2] == 10 and not c[6] for c in ineligible)
    # 比较发生在 t = 10, 20, 30, …（每次换计划后 j 从 0 数，j=10 只比较不换，j=20 换）
    assert [c[1] for c in ctrl.checks] == list(range(10, 200, 10))


def test_min_gap_with_large_m_no_check_before_gap():
    # m=30, min_gap=50, tau=0：j=30 不能换，j=60 换 → 间隔 60
    ctrl, _, _ = run(dict(m=30, tau=0, min_gap=50), B=1, T=300)
    assert [t for e, t, c in ctrl.replans] == [0, 60, 120, 180, 240]


# ---------------------------------------------------------------- 块用完
def test_exhaust_when_no_trigger_and_m_not_dividing_L():
    # m=30：比较 j=30, 60, 90（j=90 只比重叠的 10 步），然后 j=100 块用完
    ctrl, acts, calls = run(dict(m=30, tau="inf"), B=1, T=250)
    assert [c[2] for c in ctrl.checks[:3]] == [30, 60, 90]
    assert [c[4] for c in ctrl.checks[:3]] == [20, 20, 10]
    assert [(t, c) for e, t, c in ctrl.replans] == [(0, CAUSE_INITIAL), (100, CAUSE_EXHAUST), (200, CAUSE_EXHAUST)]
    assert acts[100, 0, 0] == np.float32(100 * 1000.0)
    assert calls[0] == 1 + 2 + ctrl.n_checks[0]


def test_exhaust_after_diff_switch_simple():
    # 常数块：t<30 预测 0，30<=t<130 预测 7，t>=130 预测 9 → 只有 t=30 的比较差异 > 0（换计划），
    # 换计划后块在 t=130 用完（从换计划那一步数 100 步，不是 t=100）
    def chunk_fn(t, B):
        v = 0.0 if t < 30 else (7.0 if t < 130 else 9.0)
        return np.full((B, L, A), v, dtype=np.float32)

    ctrl, acts, calls = run(dict(m=30, tau=1.0, min_gap=0), B=1, T=250, chunk_fn=chunk_fn)
    rp = [(t, c) for e, t, c in ctrl.replans]
    assert rp == [(0, CAUSE_INITIAL), (30, CAUSE_DIFF), (130, CAUSE_EXHAUST), (230, CAUSE_EXHAUST)]
    assert acts[29, 0, 0] == 0.0 and acts[30, 0, 0] == 7.0 and acts[129, 0, 0] == 7.0 and acts[130, 0, 0] == 9.0
    # 比较：t=30(j30,换) 60 90 120(j=90) | 160 190 220 ; 块用完 t=130, 230
    assert [c[1] for c in ctrl.checks] == [30, 60, 90, 120, 160, 190, 220]
    assert ctrl.n_diff[0] == 1 and ctrl.n_exhaust[0] == 2
    assert calls[0] == 1 + ctrl.n_exhaust[0] + ctrl.n_checks[0] == 10


# ---------------------------------------------------------------- 调用次数计数 / 已结束环境
def test_dead_env_not_counted_but_keeps_plan():
    def alive_fn(t):
        return np.array([True, t < 150])

    ctrl, acts, calls = run(dict(m=10, tau="inf"), B=2, T=400, alive_fn=alive_fn)
    # 环境 1 在第 150 步结束：只计结束前的调用（初始 + t=100 块用完 + 比较 t=10..90、110..140 共 13 次）
    assert ctrl.n_exhaust[1] == 1 and ctrl.n_checks[1] == 9 + 4
    assert calls[1] == ctrl.n_calls[1] == 1 + 1 + 13
    assert calls[0] == 1 + 3 + 36
    assert not any(e == 1 and t >= 150 for e, t, *_ in ctrl.checks)
    # 已结束环境照常按计划执行、块用完时换块（与 fixed 模式相同）
    t = np.arange(400)
    ref = (t // 100 * 100) * 1000.0 + t % 100 + 1e6
    assert np.array_equal(acts[:, 1, 0], ref.astype(np.float32))
    pj = np.stack(ctrl.plan_j)
    assert (pj[150:, 1] == -1).all() and pj[149, 1] == 49


def test_need_predict_only_when_needed():
    ctrl = AdaptiveController(2, dict(m=10, tau="inf"), L)
    alive = np.ones(2, dtype=bool)
    assert ctrl.need_predict(alive).all()                 # 第 0 步
    with pytest.raises(ValueError):
        ctrl.step(0, alive, None)
    ctrl.step(0, alive, fake_chunks(0, 2))
    for t in range(1, 10):
        assert not ctrl.need_predict(alive).any()
        ctrl.step(t, alive, None)                          # 不需要时可以不给 chunks
    assert ctrl.need_predict(alive).all()                  # j=10


def test_npz_roundtrip(tmp_path):
    ctrl, _, _ = run(dict(m=10, tau="inf"), B=2, T=120)
    p = tmp_path / "run" / "adaptive" / "batch_0000.npz"
    ctrl.write_npz(p, [5, 6], [120, 120])
    z = np.load(p)
    assert z["plan_j"].shape == (120, 2) and int(z["m"]) == 10 and math.isinf(float(z["tau"]))
    rows = load_checks(tmp_path / "run")
    assert len(rows) == ctrl.n_checks.sum() and {r["seed"] for r in rows} == {5, 6}
    assert set(z["rp_cause"].tolist()) == {CAUSE_INITIAL, CAUSE_EXHAUST}


# ---------------------------------------------------------------- 配置 / 指纹 / 汇总
def test_eval_config_adaptive():
    c = EvalConfig(model="act", seeds="0-9", mode="adaptive", adaptive=dict(m=10, tau="inf"))
    assert c.k_effective == 100 and c.adaptive == dict(m=10, tau=math.inf, h_cmp=20, min_gap=20)
    validate_k("act", "adaptive", 100, ACT)
    with pytest.raises(ValueError, match="chunk_size"):
        validate_k("act", "adaptive", 50, ACT)
    with pytest.raises(ValueError):
        EvalConfig(model="dp", seeds="0-9", mode="adaptive", adaptive=dict(m=10, tau=1))
    with pytest.raises(ValueError, match="mode=adaptive"):
        EvalConfig(model="act", seeds="0-9", mode="fixed", adaptive=dict(m=10, tau=1))
    with pytest.raises(ValueError):
        EvalConfig(model="act", seeds="0-9", mode="adaptive")   # 没给 m / tau


def test_load_config_adaptive_cli_overrides(tmp_path):
    y = tmp_path / "a.yaml"
    y.write_text("model: act\nseeds: 0-9\nmode: adaptive\nadaptive:\n  m: 20\n  tau: .inf\n", encoding="utf-8")
    c = load_config(y, dict(adaptive_params=dict(m=None, tau="0.05", min_gap=None, h_cmp=None)))
    assert c.adaptive == dict(m=20, tau=0.05, h_cmp=20, min_gap=20)
    c2 = load_config(y, dict(adaptive_params=dict(m=None, tau=None, min_gap=None, h_cmp=None)))
    assert math.isinf(c2.adaptive["tau"])


def test_fingerprint_adaptive_and_old_unchanged():
    from chunkre.runner import _fingerprint

    a = EvalConfig(model="act", seeds="0-9")
    assert "adaptive" not in _fingerprint(a)
    old = json.loads((ROOT / "results" / "main" / "k100__none" / "config.json").read_text(encoding="utf-8"))
    fp = old["eval_config_fingerprint"]
    cfg = EvalConfig(**{k: v for k, v in fp.items() if k in EvalConfig.__dataclass_fields__})
    assert _fingerprint(cfg) == fp
    b = EvalConfig(model="act", seeds="0-9", mode="adaptive", adaptive=dict(m=10, tau="inf"))
    fb = _fingerprint(b)
    assert fb["adaptive"]["m"] == 10
    # 写进 config.json（JSON 的 Infinity）再读回，指纹仍相等 → 可续跑
    assert json.loads(json.dumps(fb)) == fb


def test_summarize_rows_adaptive_columns():
    base = dict(batch_idx=0, success=1, steps=300, capped=0, sum_reward=1, max_reward=4, n_policy_calls=30,
                smoothness=0.01, batch_wall_s=10)
    rows = [dict(base, n_checks=26, n_replans_diff=1, n_replans_exhaust=2),
            dict(base, n_checks=20, n_replans_diff=3, n_replans_exhaust=0)]
    s = summarize_rows(rows)
    assert s["mean_n_checks"] == 23 and s["mean_n_replans_diff"] == 2 and s["mean_n_replans_exhaust"] == 1
    s0 = summarize_rows([base])
    assert not any(f"mean_{c}" in s0 for c in ADAPTIVE_COLUMNS)
