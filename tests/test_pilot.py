"""T6a 试跑：网格定义、配置生成、完成判断、汇总表 / 图（用合成的 episodes.csv，不跑仿真）。"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from chunkre.config import load_config
from chunkre.pilot import CSV_COLUMNS, PILOT_SEEDS, cell_done, cells, config_yaml, flag_of, write_configs

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_grid_is_55_plus_noise32_supplement_unique():
    cs = cells()
    assert len(cs) == 59 and len({c.name for c in cs}) == 59
    assert {c.policy for c in cs} == {"k10", "k20", "k50", "k100", "te"}
    by_kind = {}
    for c in cs:
        by_kind.setdefault(c.kind, set()).add(c.strength)
    assert by_kind == {"none": {0.0}, "push": {0.02, 0.04, 0.06}, "obs_noise": {4.0, 8.0, 16.0, 32.0},
                       "action_delay": {1.0, 3.0, 5.0, 10.0}}
    n32 = [c.name for c in cs if c.kind == "obs_noise" and c.strength == 32.0]
    assert n32 == ["k20__noise32", "k50__noise32", "k100__noise32", "te__noise32"]   # T6b 补测：k=10 不测
    assert [c.name for c in cs][:5] == ["k10__none", "k20__none", "k50__none", "k100__none", "te__none"]


def test_seeds_are_tuning_range_5000_5019():
    from chunkre.config import parse_seeds
    s = parse_seeds(PILOT_SEEDS)
    assert s[0] == 5000 and s[-1] == 5019 and len(s) == 20


def test_generated_configs_load_and_match_cell(tmp_path):
    paths = write_configs(tmp_path)
    assert len(paths) == 59
    for c, p in zip(cells(), paths):
        cfg = load_config(str(p))
        assert cfg.model == "act" and cfg.batch_size == 10 and len(cfg.seeds) == 20
        assert cfg.seeds[0] == 5000 and cfg.seeds[-1] == 5019
        assert cfg.name == c.name == p.stem
        assert cfg.mode == c.mode and cfg.k_effective == (1 if c.mode == "te" else c.k)
        assert cfg.perturb["kind"] == c.kind and cfg.perturb["strength"] == c.strength
        if c.kind == "push":
            assert cfg.perturb["params"]["t_lo"] == 20 and cfg.perturb["params"]["t_hi"] == 100
    assert "perturb" not in config_yaml(cells()[0])
    assert b"\r" not in paths[0].read_bytes()


def test_committed_configs_are_up_to_date(tmp_path):
    committed = ROOT / "configs" / "pilot"
    if not committed.exists():
        pytest.skip("configs/pilot 还没生成")
    write_configs(tmp_path)
    assert sorted(p.name for p in committed.glob("*.yaml")) == sorted(p.name for p in tmp_path.glob("*.yaml"))
    for p in tmp_path.glob("*.yaml"):
        assert (committed / p.name).read_bytes() == p.read_bytes(), p.name


def test_flag_thresholds():
    assert flag_of(0.0) == "floor" and flag_of(0.10) == "floor" and flag_of(0.101) == ""
    assert flag_of(0.9) == "ceiling" and flag_of(0.899) == "" and flag_of(float("nan")) == ""


def _write_cell(root: Path, name: str, n: int, n_success: int, summary=True):
    d = root / name
    d.mkdir(parents=True)
    with open(d / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "success", "steps", "n_policy_calls", "path_length", "jerk"])
        w.writeheader()
        for i in range(n):
            w.writerow(dict(seed=5000 + i, success=int(i < n_success), steps=300 + i, n_policy_calls=4 + i % 2,
                            path_length=10.0 + i, jerk=0.01))
    if summary:
        (d / "summary.json").write_text(json.dumps({}), encoding="utf-8")
    return d


def test_cell_done(tmp_path):
    d = _write_cell(tmp_path, "a", 20, 5)
    assert cell_done(d, 20) and not cell_done(d, 21)
    assert not cell_done(_write_cell(tmp_path, "b", 20, 5, summary=False), 20)   # 没有 summary.json = 没跑完
    assert not cell_done(tmp_path / "nonexistent", 20)


def test_analyze_table_and_figures(tmp_path):
    pytest.importorskip("matplotlib")
    ap = _load_script("analyze_pilot")
    _write_cell(tmp_path, "k10__none", 20, 2)        # 地板（2/20 = 10%）
    _write_cell(tmp_path, "k100__none", 20, 18)      # 天花板
    _write_cell(tmp_path, "k100__noise8", 20, 10)
    _write_cell(tmp_path, "te__delay3", 7, 3)        # 回合数不足
    stats, missing = ap.collect(tmp_path)
    by = {s["cell"]: s for s in stats}
    assert len(stats) == 4 and len(missing) == 55
    assert by["k10__none"]["flag"] == "floor" and by["k100__none"]["flag"] == "ceiling"
    assert by["k100__noise8"]["flag"] == "" and by["k100__noise8"]["success_rate"] == 0.5
    lo, hi = by["k100__noise8"]["ci_lo"], by["k100__noise8"]["ci_hi"]
    assert 0.29 < lo < 0.30 and 0.70 < hi < 0.71          # Wilson(10/20)
    assert by["k100__noise8"]["mean_policy_calls"] == 4.5 and by["k100__noise8"]["mean_path_length"] == 19.5
    md = ap.build_markdown(stats, missing, 20)
    assert "▼floor" in md and "▲ceil" in md and "te__delay3（7）" in md and "显著" not in md
    ap.write_csv(stats, tmp_path / "pilot_table.csv")
    with open(tmp_path / "pilot_table.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == CSV_COLUMNS and len(rows) == 4
    figs = ap.plot_figures(stats, tmp_path)
    assert sorted(p.name for p in figs) == ["fig_delay.png", "fig_noise.png", "fig_push.png"]
    assert all(p.stat().st_size > 5000 for p in figs)


def test_analyze_main_no_cells(tmp_path):
    ap = _load_script("analyze_pilot")
    with pytest.raises(SystemExit):
        ap.main(["--dir", str(tmp_path)])
