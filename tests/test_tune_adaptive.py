"""T11a 阈值试跑网格与选择规则（src/chunkre/tune_adaptive.py）。不跑仿真。"""
import json
import math
from pathlib import Path

import pytest
import yaml

from chunkre import tune_adaptive as ta
from chunkre.config import EvalConfig, parse_seeds

ROOT = Path(__file__).resolve().parent.parent
CHECKS = ROOT / "results" / "t11_checks" / "checks.json"
FAKE_TAUS = (0.01, 0.02, 0.03, 0.04, 0.05)


@pytest.fixture
def taus(monkeypatch):
    monkeypatch.setattr(ta, "TAU_CANDIDATES", FAKE_TAUS)


def test_tau_candidates_match_a3():
    """写进代码的 tau 候选值 = A3 无扰动分布分位数（results/t11_checks/checks.json）。"""
    a3 = json.loads(CHECKS.read_text(encoding="utf-8"))["a3"]
    assert list(ta.TAU_CANDIDATES) == a3["tau_candidates"]
    assert list(ta.TAU_CANDIDATES) == [round(a3["none"][f"p{q}"], 4) for q in (50, 75, 90, 95, 99)]
    assert list(ta.TAU_CANDIDATES) == sorted(ta.TAU_CANDIDATES)


def test_seeds_new_tuning_range():
    s = parse_seeds(ta.TUNE_SEEDS)
    assert s == list(range(6000, 6050)) and len(s) == ta.TUNE_N and max(s) < 10000
    # 此前用过的调参种子：0–19（检查）、1000–1499（复现）、5000–5049（试跑）
    assert not set(s) & (set(range(0, 20)) | set(range(1000, 1500)) | set(range(5000, 5050)))


def test_grid_22_cells(taus):
    cs = ta.cells()
    assert len(cs) == 22 and len({c.name for c in cs}) == 22
    assert [c.name for c in cs[:2]] == ["k100__none", "k100__push0.06"]
    ad = [c for c in cs if c.adaptive]
    assert len(ad) == 20 and {c.m for c in ad} == {10, 20} and {c.tau for c in ad} == set(FAKE_TAUS)
    assert "ad_m20_q99__push0.06" in {c.name for c in cs}


def test_config_yaml_loads(taus, tmp_path):
    for c in ta.cells():
        d = yaml.safe_load(ta.config_yaml(c))
        cfg = EvalConfig(**d)
        assert cfg.seeds == list(range(6000, 6050)) and cfg.batch_size == 10 and cfg.k_effective == 100
        if c.adaptive:
            assert cfg.mode == "adaptive" and cfg.adaptive == dict(m=c.m, tau=c.tau, h_cmp=20, min_gap=20)
        else:
            assert cfg.mode == "fixed" and cfg.k == 100 and cfg.adaptive == {}
        if c.cond == "none":
            assert cfg.perturb["kind"] == "none"
        else:
            assert cfg.perturb["kind"] == "push" and cfg.perturb["strength"] == 0.06
            assert cfg.perturb["params"]["t_lo"] == 20 and cfg.perturb["params"]["t_hi"] == 100
    d = yaml.safe_load(ta.config_yaml(ta.cells()[3], seeds="6000-6001", batch_size=2, max_steps=40))
    assert EvalConfig(**d).max_steps == 40


def stats(ref_none=40, ref_push=10, n=50, combos=None, default=(30, 5, 20.0)):
    """combos: {组合名: (无扰动成功, 推成功, 平均调用)}；其余组合用 default。"""
    out = {"k100__none": dict(n=n, n_success=ref_none, mean_policy_calls=3.6),
           "k100__push0.06": dict(n=n, n_success=ref_push, mean_policy_calls=3.8)}
    for m, q, _ in ta.combos():
        cn = ta.combo_name(m, q)
        sn, sp, calls = (combos or {}).get(cn, default)
        out[f"{cn}__none"] = dict(n=n, n_success=sn, mean_policy_calls=calls)
        out[f"{cn}__push0.06"] = dict(n=n, n_success=sp, mean_policy_calls=calls)
    return out


def test_select_best_push_among_eligible(taus):
    # 门槛 = 40/50 − 3pp = 77%：38/50 = 76% 不可用，39/50 = 78% 可用
    s = stats(combos={"ad_m10_q50": (38, 30, 20.0), "ad_m10_q90": (39, 25, 20.0), "ad_m20_q75": (45, 20, 10.0)})
    r = ta.select(s)
    assert r["verdict"] == "ad_m10_q90" and r["n_eligible"] == 2 and not r["still_tied"]
    assert math.isclose(r["none_bar"], 0.77)


def test_select_boundary_exact_3pp(taus):
    # 参照 50/50 → 门槛 97%；n=100 时 97/100 正好在门槛上（可用），96/100 不可用
    s = stats(ref_none=100, n=100, default=(96, 50, 20.0), combos={"ad_m20_q99": (97, 10, 20.0)})
    r = ta.select(s)
    assert r["verdict"] == "ad_m20_q99" and r["n_eligible"] == 1


def test_select_tie_broken_by_calls(taus):
    s = stats(default=(45, 20, 30.0), combos={"ad_m10_q75": (45, 25, 30.0), "ad_m20_q50": (45, 25, 12.5)})
    r = ta.select(s)
    assert r["verdict"] == "ad_m20_q50" and r["tied_on_push"] == ["ad_m10_q75", "ad_m20_q50"]


def test_select_still_tied_takes_grid_order(taus):
    s = stats(default=(45, 20, 30.0), combos={"ad_m10_q99": (45, 25, 12.0), "ad_m20_q50": (45, 25, 12.0)})
    r = ta.select(s)
    assert r["verdict"] == "ad_m10_q99" and r["still_tied"] and r["tied_with"] == ["ad_m20_q50"]


def test_select_none_eligible(taus):
    r = ta.select(stats(ref_none=50, default=(40, 49, 5.0)))
    assert r["selected"] is None and r["verdict"] == "无可用组合" and r["n_eligible"] == 0


def test_select_missing_cell(taus):
    s = stats()
    s.pop("ad_m10_q50__push0.06")
    with pytest.raises(ValueError, match="缺少"):
        ta.select(s)
