"""T8b DP 试跑：网格定义、配置生成、汇总表 / 图（用合成的 episodes.csv，不跑仿真）。"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from chunkre.config import load_config, parse_seeds
from chunkre.perturb import PushTee
from chunkre.pilot_dp import (CSV_COLUMNS, PILOT_DP_SEEDS, PUSH_PARAMS, cells, config_yaml, write_configs)
from chunkre.seed_guard import check_seeds

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_grid_is_4x10_unique():
    cs = cells()
    assert len(cs) == 40 and len({c.name for c in cs}) == 40
    assert [c.k for c in cs[:4]] == [2, 4, 8, 15] and {c.mode for c in cs} == {"fixed"}
    by_kind = {}
    for c in cs:
        by_kind.setdefault(c.kind, set()).add(c.strength)
    assert by_kind == {"none": {0.0}, "push": {20.0, 40.0, 60.0}, "obs_noise": {4.0, 8.0, 16.0},
                       "action_delay": {1.0, 2.0, 4.0}}
    assert [c.name for c in cs][:4] == ["k2__none", "k4__none", "k8__none", "k15__none"]


def test_seeds_5000_5049_in_tuning_range():
    s = parse_seeds(PILOT_DP_SEEDS)
    assert s[0] == 5000 and s[-1] == 5049 and len(s) == 50 and max(s) < 10000
    check_seeds(s)


def test_push_params_match_pushtee_defaults():
    assert PUSH_PARAMS == {k: PushTee.DEFAULTS[k] for k in ("t_lo", "t_hi")}


def test_generated_configs_load_and_match_cell(tmp_path):
    paths = write_configs(tmp_path)
    assert len(paths) == 40
    for c, p in zip(cells(), paths):
        cfg = load_config(str(p))
        assert cfg.model == "dp" and cfg.batch_size == 50 and len(cfg.seeds) == 50
        assert cfg.name == c.name == p.stem and cfg.k_effective == c.k and cfg.k == c.k
        assert cfg.perturb["kind"] == c.kind and cfg.perturb["strength"] == c.strength
        if c.kind == "push":
            assert cfg.perturb["params"] == PushTee.DEFAULTS
    assert "perturb" not in config_yaml(cells()[0])
    assert b"\r" not in paths[0].read_bytes()


def test_committed_configs_are_up_to_date(tmp_path):
    committed = ROOT / "configs" / "pilot_dp"
    if not committed.exists():
        pytest.skip("configs/pilot_dp 还没生成")
    write_configs(tmp_path)
    assert sorted(p.name for p in committed.glob("*.yaml")) == sorted(p.name for p in tmp_path.glob("*.yaml"))
    for p in tmp_path.glob("*.yaml"):
        assert (committed / p.name).read_bytes() == p.read_bytes(), p.name


def _write_cell(root: Path, name: str, n: int, n_success: int):
    d = root / name
    d.mkdir(parents=True)
    with open(d / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "success", "steps", "n_policy_calls", "path_length", "jerk",
                                          "max_reward", "perturb_kind", "push_status", "push_clip"])
        w.writeheader()
        push = "__push" in name
        for i in range(n):
            w.writerow(dict(seed=5000 + i, success=int(i < n_success), steps=200 + i, n_policy_calls=30 + i % 2,
                            path_length=1000.0 + i, jerk=4.0, max_reward=1.0 if i < n_success else 0.5,
                            perturb_kind="push" if push else "none",
                            push_status=(["pushed", "skipped_overlap", "not_reached"][i % 3] if push else ""),
                            push_clip=("wall" if push and i == 0 else "")))
    (d / "summary.json").write_text(json.dumps({}), encoding="utf-8")


def test_analyze_dp_grid(tmp_path):
    pytest.importorskip("matplotlib")
    ap = _load_script("analyze_pilot")
    _write_cell(tmp_path, "k8__none", 50, 33)
    _write_cell(tmp_path, "k2__push60", 50, 5)        # 地板（10%）
    _write_cell(tmp_path, "k15__delay4", 20, 10)      # 回合数不足
    stats, missing = ap.collect(tmp_path, "dp")
    by = {s["cell"]: s for s in stats}
    assert len(stats) == 3 and len(missing) == 37
    assert by["k8__none"]["success_rate"] == 0.66 and by["k2__push60"]["flag"] == "floor"
    assert by["k8__none"]["mean_max_reward"] == pytest.approx((33 * 1.0 + 17 * 0.5) / 50)
    assert by["k8__none"]["mean_policy_calls"] == pytest.approx(30.5)
    md = ap.build_markdown(stats, missing, 50, "dp")
    assert "T8b" in md and "平均最大覆盖率" in md and "k15__delay4（20）" in md and "显著" not in md
    assert "| push60 |" in md and "| delay4 |" in md
    assert by["k2__push60"]["n_pushed"] == 17 and by["k2__push60"]["n_skipped_overlap"] == 17
    assert by["k2__push60"]["n_not_reached"] == 16 and by["k2__push60"]["n_clip_wall"] == 1
    assert by["k8__none"]["n_pushed"] == 0
    assert "| k2__push60 | 17 | 17 | 16 | 1 |" in md
    ap.write_csv(stats, tmp_path / "pilot_table.csv", "dp")
    with open(tmp_path / "pilot_table.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == CSV_COLUMNS and len(rows) == 3
    figs = ap.plot_figures(stats, tmp_path, "dp")
    assert sorted(p.name for p in figs) == ["fig_delay.png", "fig_noise.png", "fig_push.png"]
    assert all(p.stat().st_size > 5000 for p in figs)


def test_eta_dp_runs_and_is_plausible():
    pt = _load_script("pilot_tools")
    s = pt.eta_seconds(cells(), 50, 50, "dp")
    assert 1.5 * 3600 < s < 4 * 3600         # T2 耗时表：k2/4/8/15 每回合约 8.1/4.2/2.4/1.4 s（B=50）
