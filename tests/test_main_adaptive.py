"""T11b：自适应正式评估的配置、与登记文件 / 试跑配置的一致性、参数核对、推之后反应步数、analyze_main_adaptive 端到端（合成数据，不跑仿真）。"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from chunkre import main_adaptive as ma
from chunkre import tune_adaptive as ta
from chunkre.config import load_config, parse_seeds
from chunkre.stats import mcnemar_exact, paired_bootstrap_diff

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name):
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_yaml_text(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return load_config(p)


# ---------- 参数 = 登记文件 = 试跑配置里 q90 的 tau ----------
def test_registered_params_parse_and_match_constants():
    assert ma.registered_params(ROOT) == ma.ADAPTIVE_PARAMS == dict(m=10, tau=0.2906, h_cmp=20, min_gap=20)


def test_tau_equals_tune_config_q90_exactly():
    q90 = dict(zip(ta.TAU_QUANTILES, ta.TAU_CANDIDATES))["q90"]
    assert ma.ADAPTIVE_PARAMS["tau"] == q90
    for tag in ("none", "push0.06"):
        cfg = load_config(ROOT / "configs" / "tune_adaptive" / f"ad_m10_q90__{tag}.yaml")
        assert cfg.adaptive == ma.ADAPTIVE_PARAMS


def test_cells_and_configs_load_and_match_registration(tmp_path):
    assert ma.cell_names() == ["ad__none", "ad__push0.06"] and ma.control_names() == ["k100__none", "k100__push0.06"]
    assert ma.ADAPTIVE_SEEDS == "200000-200199" and ma.ADAPTIVE_N == 200 and ma.ADAPTIVE_BATCH == 10
    for tag, kind, strength in ma.CONDITIONS:
        cfg = _load_yaml_text(tmp_path, ma.config_yaml(tag))
        assert cfg.model == "act" and cfg.mode == "adaptive" and cfg.adaptive == ma.ADAPTIVE_PARAMS
        assert cfg.policy_path is None and cfg.batch_size == 10 and len(cfg.seeds) == 200 and cfg.seeds[0] == 200000


def test_push_config_same_as_main_experiment(tmp_path):
    """推的参数与正式实验 k100__push0.06 的 config.json 里的扰动设置相同。"""
    main = json.loads((ROOT / "results" / "main" / "k100__push0.06" / "config.json").read_text(encoding="utf-8"))["perturbation"]
    cfg = _load_yaml_text(tmp_path, ma.config_yaml("push0.06"))
    pert = cfg.to_dict()["perturb"]
    assert pert["kind"] == main["kind"] and float(pert["strength"]) == main["strength"] and dict(pert["params"]) == main["params"]


def test_control_cells_same_seeds_as_adaptive():
    for n in ma.control_names():
        fp = json.loads((ROOT / "results" / "main" / n / "config.json").read_text(encoding="utf-8"))["eval_config_fingerprint"]
        assert fp["seeds"] == parse_seeds(ma.ADAPTIVE_SEEDS) and fp["mode"] == "fixed" and fp["k"] == 100


def test_check_run_params_detects_each_difference(tmp_path):
    reg = ma.registered_params(ROOT)
    ps = ma.write_configs(tmp_path / "ok")
    assert ma.check_run_params(ps, reg) == []
    for key, val in (("m", 20), ("tau", 0.3), ("tau", 0.29061), ("h_cmp", 10), ("min_gap", 10)):
        params = dict(ma.ADAPTIVE_PARAMS, **{key: val})
        bad = ma.check_run_params(ma.write_configs(tmp_path / f"bad_{key}_{val}", params), reg)
        assert len(bad) == 2 and all(key in b for b in bad)    # 两个配置都报
    p = tmp_path / "fixed.yaml"
    p.write_text("name: x\nmodel: act\nseeds: 1-2\nmode: fixed\nk: 100\n", encoding="utf-8")
    assert "adaptive" in ma.check_run_params([p], reg)[0]


def test_tools_verify_params_exit_codes(tmp_path, capsys):
    tools = _load_script("main_adaptive_tools")
    cfg = tmp_path / "cfg"
    tools.main(["gen", "--out", str(cfg)])
    tools.main(["verify-params", "--out", str(cfg)])           # 一致：不退出
    tools.main(["gen", "--out", str(cfg), "--tau", "0.3"])
    with pytest.raises(SystemExit) as e:
        tools.main(["verify-params", "--out", str(cfg)])
    assert e.value.code == 4
    assert "tau" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        tools.main(["verify-params", "--out", str(tmp_path / "nonexistent")])
    assert e.value.code == 4


def test_tools_guard_passes_in_repo():
    tools = _load_script("main_adaptive_tools")
    from chunkre.seed_guard import preregistration_committed
    if not preregistration_committed(model="act_adaptive"):
        pytest.skip("没有 git 历史")
    tools.main(["guard"])


# ---------- 推之后的第一次差异换计划 ----------
def test_first_reaction_steps_rules():
    rows = [dict(seed=1, push_status="pushed", push_step=50),     # 反应在 t=65 -> 15
            dict(seed=2, push_status="pushed", push_step=30),     # 差异换计划只有 t=30（推之前的画面）和块用完 -> 无反应
            dict(seed=3, push_status="skipped_lifted", push_step=40),   # 没推：不算
            dict(seed=4, push_status="pushed", push_step=20),     # 无任何记录 -> 无反应
            dict(seed=5, push_status="pushed", push_step=60)]     # t=60（同一步，不算）后 t=70 -> 10
    rp = [dict(seed=1, t=0, cause=0), dict(seed=1, t=40, cause=2), dict(seed=1, t=65, cause=2), dict(seed=1, t=90, cause=2),
          dict(seed=2, t=30, cause=2), dict(seed=2, t=100, cause=1),
          dict(seed=3, t=45, cause=2),
          dict(seed=5, t=60, cause=2), dict(seed=5, t=70, cause=2)]
    r = ma.first_reaction_steps(rows, rp)
    assert r["steps"] == [15, 10] and r["n_pushed"] == 4 and r["n_no_reaction"] == 2 and r["n_replan_at_push_step"] == 2


def test_load_replans_roundtrip(tmp_path):
    d = tmp_path / "cell" / "adaptive"
    d.mkdir(parents=True)
    np.savez(d / "batch_0000.npz", seeds=np.array([7, 8]), rp_env=np.array([0, 1, 1], dtype=np.int16),
             rp_t=np.array([0, 0, 33], dtype=np.int16), rp_cause=np.array([0, 0, 2], dtype=np.int8))
    assert ma.load_replans(tmp_path / "cell") == [dict(seed=7, t=0, cause=0), dict(seed=8, t=0, cause=0), dict(seed=8, t=33, cause=2)]


# ---------- analyze_main_adaptive 端到端（合成数据） ----------
COLS = ["batch_idx", "env_idx", "seed", "success", "n_policy_calls", "jerk", "path_length", "smoothness", "push_step", "push_status",
        "push_clip", "n_checks", "n_replans_diff", "n_replans_exhaust"]


def _write_cell(root, name, seeds, success, calls=3.0, adaptive=False, push=False, params=None):
    d = Path(root) / name
    d.mkdir(parents=True)
    with open(d / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for i, (s, ok) in enumerate(zip(seeds, success)):
            w.writerow(dict(batch_idx=i // 10, env_idx=i % 10, seed=s, success=int(ok), n_policy_calls=calls, jerk=0.02, path_length=8.0,
                            smoothness=0.02, push_step=50 if push else "", push_status="pushed" if push else "", push_clip="",
                            n_checks=30 if adaptive else "", n_replans_diff=(i % 3) if adaptive else "",
                            n_replans_exhaust=1 if adaptive else ""))
    fp = dict(model="act", mode="adaptive" if adaptive else "fixed", k=None if adaptive else 100,
              adaptive=dict(params or ma.ADAPTIVE_PARAMS) if adaptive else {})
    (d / "config.json").write_text(json.dumps({"eval_config_fingerprint": fp}), encoding="utf-8")
    if adaptive and push:   # 换计划记录：偶数种子在推后 12 步（t=62，推在 50）反应，其余没有
        (d / "adaptive").mkdir()
        for b in range(len(seeds) // 10):
            ss = list(seeds[b * 10:(b + 1) * 10])
            envs = [i for i, s in enumerate(ss) if s % 2 == 0]
            np.savez(d / "adaptive" / f"batch_{b:04d}.npz", seeds=np.array(ss), rp_env=np.array(envs, dtype=np.int16),
                     rp_t=np.full(len(envs), 62, dtype=np.int16), rp_cause=np.full(len(envs), 2, dtype=np.int8))
    return d


def _synthetic(tmp_path, seeds_start, ad_none, ad_push, k_none, k_push, params=None):
    n = 200
    seeds = list(range(seeds_start, seeds_start + n))
    main = tmp_path / "main"
    out = tmp_path / "ad"
    _write_cell(out, "ad__none", seeds, ad_none, 33.0, adaptive=True, params=params)
    _write_cell(out, "ad__push0.06", seeds, ad_push, 36.0, adaptive=True, push=True, params=params)
    _write_cell(main, "k100__none", seeds, k_none)
    _write_cell(main, "k100__push0.06", seeds, k_push, push=True)
    for pol in ("k20", "k50", "te"):
        for tag in ("none", "push0.06"):
            _write_cell(main, f"{pol}__{tag}", seeds, [i % 2 for i in range(n)], 7.0)
    return seeds, out, main


def test_analyze_end_to_end_formal_like(tmp_path):
    """合成数据：自适应推 0.06 明显更好、无扰动与 k=100 相同 -> HA1 supported，HA2 holds，结论“有用”；参数与登记相同、种子 >= 200000 -> 不是 DRY RUN。"""
    mod = _load_script("analyze_main_adaptive")
    rng = np.random.default_rng(0)
    k_none = (rng.random(200) < 0.78).astype(int)
    ad_none = k_none.copy()
    k_push = (rng.random(200) < 0.18).astype(int)
    ad_push = k_push.copy()
    ad_push[np.where(k_push == 0)[0][:40]] = 1                 # 40 个只有自适应成功
    seeds, out, main = _synthetic(tmp_path, 200000, ad_none, ad_push, k_none, k_push)
    mod.main(["--dir", str(out), "--control-dir", str(main), "--out", str(tmp_path / "rep")])
    rep = (tmp_path / "rep" / "main_adaptive_report.md").read_text(encoding="utf-8")
    js = json.loads((tmp_path / "rep" / "hypotheses_adaptive.json").read_text(encoding="utf-8"))
    assert "DRY RUN" not in rep and js["dry_run"] is False
    h1, h2 = js["HA1"], js["HA2"]
    assert h1["only_a"] == 40 and h1["only_b"] == 0 and h1["p_mcnemar"] == mcnemar_exact(40, 0) and h1["decision"] == "supported"
    d, lo, hi = paired_bootstrap_diff(ad_none, k_none)
    assert h2["diff"] == pytest.approx(0.0) and h2["ci_lo"] == pytest.approx(lo) and h2["holds"] is True
    assert js["conclusion"]["useful"] is True
    assert (tmp_path / "rep" / "fig_adaptive.png").exists() and (tmp_path / "rep" / "main_adaptive_cells.csv").exists()
    assert "cheaper fixed control" in rep and "k=50" in rep
    rc = js["reaction"]
    assert rc["available"] and rc["n_with_reaction"] == 100 and rc["median"] == 12.0 and rc["n_no_reaction"] == 100


def test_analyze_noninferiority_fails_and_opposite(tmp_path):
    mod = _load_script("analyze_main_adaptive")
    rng = np.random.default_rng(1)
    k_none = (rng.random(200) < 0.8).astype(int)
    ad_none = k_none.copy()
    ad_none[np.where(k_none == 1)[0][:30]] = 0                    # 30 个只有 k=100 成功：差 −0.15，下界明显 < −0.05
    k_push = (rng.random(200) < 0.5).astype(int)
    ad_push = k_push.copy()
    ad_push[np.where(k_push == 1)[0][:30]] = 0                    # 自适应更差：相反结果
    seeds, out, main = _synthetic(tmp_path, 200000, ad_none, ad_push, k_none, k_push)
    mod.main(["--dir", str(out), "--control-dir", str(main), "--out", str(tmp_path / "rep")])
    js = json.loads((tmp_path / "rep" / "hypotheses_adaptive.json").read_text(encoding="utf-8"))
    assert js["HA1"]["decision"] == "opposite" and js["HA2"]["holds"] is False and js["conclusion"]["useful"] is False
    assert "OPPOSITE RESULT" in (tmp_path / "rep" / "main_adaptive_report.md").read_text(encoding="utf-8")


def test_analyze_dry_run_marked_on_tuning_seeds_and_wrong_params(tmp_path):
    mod = _load_script("analyze_main_adaptive")
    ones = np.ones(200, dtype=int)
    # 调参种子 -> DRY RUN
    seeds, out, main = _synthetic(tmp_path / "a", 6000, ones, ones, ones, ones)
    mod.main(["--dir", str(out), "--control-dir", str(main), "--seeds", "6000-6199", "--out", str(tmp_path / "a" / "rep")])
    assert "DRY RUN" in (tmp_path / "a" / "rep" / "main_adaptive_report.md").read_text(encoding="utf-8")
    # 种子 >= 200000 但参数与登记不一致（tau=0.3）-> 也是 DRY RUN
    seeds, out, main = _synthetic(tmp_path / "b", 200000, ones, ones, ones, ones, params=dict(ma.ADAPTIVE_PARAMS, tau=0.3))
    mod.main(["--dir", str(out), "--control-dir", str(main), "--out", str(tmp_path / "b" / "rep")])
    rep = (tmp_path / "b" / "rep" / "main_adaptive_report.md").read_text(encoding="utf-8")
    assert "DRY RUN" in rep and "tau = 0.3" in rep
    # 对照少一个种子 -> 涉及它的检验不做
    seeds, out, main = _synthetic(tmp_path / "c", 200000, ones, ones, ones, ones)
    f = main / "k100__none" / "episodes.csv"
    f.write_text("\n".join(f.read_text().splitlines()[:-1]) + "\n")
    mod.main(["--dir", str(out), "--control-dir", str(main), "--out", str(tmp_path / "c" / "rep")])
    js = json.loads((tmp_path / "c" / "rep" / "hypotheses_adaptive.json").read_text(encoding="utf-8"))
    assert js["HA2"]["available"] is False and js["HA1"]["available"] is True
