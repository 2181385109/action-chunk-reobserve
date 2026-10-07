"""T6b：正式网格、配置生成、子集选择、run_main 辅助、analyze_main 端到端（合成的 episodes.csv，不跑仿真）。"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from chunkre import main_grid as mg
from chunkre.config import load_config, parse_seeds

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name):
    if "scripts_dir" not in sys.modules:
        sys.path.insert(0, str(ROOT / "scripts"))
        sys.modules["scripts_dir"] = True
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- 网格 ----------
def test_grid_is_4x10_unique():
    cs = mg.cells()
    assert len(cs) == 40 and len({c.name for c in cs}) == 40
    assert {c.policy for c in cs} == {"k20", "k50", "k100", "te"}
    assert len(mg.condition_tags()) == 10 and mg.condition_tags()[0] == "none"
    by_kind = {}
    for c in cs:
        by_kind.setdefault(c.kind, set()).add(c.strength)
    assert by_kind == {"none": {0.0}, "push": {0.02, 0.04, 0.06}, "obs_noise": {float(s) for s in mg.NOISE_LEVELS},
                       "action_delay": {3.0, 5.0, 10.0}}
    assert len(mg.NOISE_LEVELS) == 3 and mg.NOISE_LEVELS == sorted(mg.NOISE_LEVELS)
    assert mg.NOISE_LEVELS in ([4, 8, 16], [8, 16, 32])    # 事先登记的两个候选，由 σ=32 补测的机械规则选定
    assert [c.name for c in cs][:4] == ["k20__none", "k50__none", "k100__none", "te__none"]


def test_main_cells_are_pilot_cells():
    """正式格子名必须都是试跑格子名（同一份分析代码才能读试跑数据做演示运行）。"""
    from chunkre.pilot import cells as pilot_cells
    assert {c.name for c in mg.cells()} <= {c.name for c in pilot_cells()}


def test_hypotheses_match_preregistration():
    h = {x[0]: x for x in mg.HYPOTHESES}
    assert h["H1"] == ("H1", "push0.06", "k20", "k100")
    assert h["H2"] == ("H2", f"noise{max(mg.NOISE_LEVELS):g}", "k100", "te")
    assert h["H3"] == ("H3", "delay10", "k100", "k50")
    names = {c.name for c in mg.cells()}
    for _, tag, A, B in mg.HYPOTHESES:
        assert f"{A}__{tag}" in names and f"{B}__{tag}" in names


def test_main_seeds_are_final_range():
    s = parse_seeds(mg.MAIN_SEEDS)
    assert s[0] == 200000 and s[-1] == 200199 and len(s) == 200 == mg.MAIN_N


def test_generated_configs_load_and_match_cell(tmp_path):
    paths = mg.write_configs(tmp_path)
    assert len(paths) == 40
    for c, p in zip(mg.cells(), paths):
        cfg = load_config(str(p))
        assert cfg.model == "act" and cfg.batch_size == 10 and len(cfg.seeds) == 200
        assert cfg.seeds[0] == 200000 and cfg.seeds[-1] == 200199
        assert cfg.name == c.name == p.stem
        assert cfg.mode == c.mode and cfg.k_effective == (1 if c.mode == "te" else c.k)
        assert cfg.perturb["kind"] == c.kind and cfg.perturb["strength"] == c.strength
        if c.kind == "push":
            assert cfg.perturb["params"]["t_lo"] == 20 and cfg.perturb["params"]["t_hi"] == 100
    assert b"\r" not in paths[0].read_bytes()


def test_committed_main_configs_are_up_to_date(tmp_path):
    committed = ROOT / "configs" / "main"
    if not committed.exists():
        pytest.skip("configs/main 还没生成")
    mg.write_configs(tmp_path)
    assert sorted(p.name for p in committed.glob("*.yaml")) == sorted(p.name for p in tmp_path.glob("*.yaml"))
    for p in tmp_path.glob("*.yaml"):
        assert (committed / p.name).read_bytes() == p.read_bytes(), p.name


def test_parse_selection():
    names = [c.name for c in mg.cells()]
    assert mg.parse_selection(None, names) == names and mg.parse_selection("all", names) == names
    assert mg.parse_selection("1-20", names) == names[:20] and mg.parse_selection("21-40", names) == names[20:]
    assert mg.parse_selection("5", names) == [names[4]]
    assert mg.parse_selection("k20__none,te__none", names) == ["k20__none", "te__none"]
    for bad in ("0-5", "30-41", "20-10", "nope__none"):
        with pytest.raises(ValueError):
            mg.parse_selection(bad, names)


def test_main_tools_split_into_two_nights_covers_all(tmp_path):
    mt = _load_script("main_tools")
    names = [c.name for c in mg.cells()]
    assert mt.pilot_batch_seconds is not None
    first, second = mg.parse_selection("1-20", names), mg.parse_selection("21-40", names)
    assert first + second == names and not set(first) & set(second)


# ---------- seed_guard：正式种子在没有 PREREGISTRATION.md 的目录里被拦住 ----------
def test_final_seeds_blocked_without_committed_preregistration(tmp_path):
    from chunkre.seed_guard import check_seeds
    with pytest.raises(RuntimeError, match="PREREGISTRATION"):
        check_seeds(parse_seeds(mg.MAIN_SEEDS), root=tmp_path)


# ---------- analyze_main ----------
def _write_cell(root: Path, name: str, seeds, success, rng, push_status=None, clip=None):
    d = root / name
    d.mkdir(parents=True)
    with open(d / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "success", "n_policy_calls", "path_length", "jerk", "smoothness",
                                          "push_status", "push_clip"])
        w.writeheader()
        for i, s in enumerate(seeds):
            w.writerow(dict(seed=s, success=int(success[i]), n_policy_calls=4 + int(rng.integers(0, 3)),
                            path_length=8 + float(rng.uniform(0, 2)), jerk=0.02, smoothness=0.02,
                            push_status=(push_status[i] if push_status else ""), push_clip=(clip[i] if clip else "")))


def _synthetic_main(root: Path, seeds, designed=True, rng_seed=0):
    """全部 40 格；H1/H2/H3 涉及的 6 格按手算设计：
    H1 (k20 vs k100 @ push0.06)：n=20，both=5, only_a=10, only_b=0, neither=5 -> p=2/1024；
    H2 (k100 vs te @ 最强噪声)：only_a=3, only_b=3 -> p=1；
    H3 (k100 vs k50 @ delay10)：only_a=0, only_b=7 -> p=2/128（方向相反）。
    Holm：排序 p = 2/1024, 2/128, 1 -> 校正 3·(2/1024)=0.005859, 2·(2/128)=0.03125, 1。"""
    rng = np.random.default_rng(rng_seed)
    n = len(seeds)
    spec = {}
    if designed:
        z = np.zeros(n, dtype=int)
        a1, b1 = z.copy(), z.copy()
        a1[:15] = 1
        b1[:5] = 1
        a2, b2 = z.copy(), z.copy()
        a2[:3] = 1
        b2[3:6] = 1
        a3, b3 = z.copy(), z.copy()
        b3[:7] = 1
        n_tag = f"noise{max(mg.NOISE_LEVELS):g}"
        spec = {"k20__push0.06": a1, "k100__push0.06": b1, f"k100__{n_tag}": a2, f"te__{n_tag}": b2,
                "k100__delay10": a3, "k50__delay10": b3}
    for c in mg.cells():
        succ = spec.get(c.name, (rng.uniform(size=n) < 0.5).astype(int))
        if c.kind == "push":
            status = ["pushed"] * n
            status[0], status[1] = "skipped_lifted", "not_reached"
            clip = [""] * n
            clip[2], clip[3] = "table", "arm"
            _write_cell(root, c.name, seeds, succ, rng, status, clip)
        else:
            _write_cell(root, c.name, seeds, succ, rng)


def test_analyze_main_end_to_end_hand_designed(tmp_path):
    pytest.importorskip("matplotlib")
    am = _load_script("analyze_main")
    seeds = list(range(5000, 5020))
    _synthetic_main(tmp_path / "data", seeds)
    out = tmp_path / "out"
    am.main(["--dir", str(tmp_path / "data"), "--out", str(out), "--seeds", "5000-5019"])
    hyp = json.loads((out / "hypotheses.json").read_text(encoding="utf-8"))
    h1, h2, h3 = hyp
    assert (h1["only_a"], h1["only_b"], h1["both"], h1["neither"]) == (10, 0, 5, 5)
    assert h1["p_raw"] == pytest.approx(2 / 1024) and h1["diff"] == pytest.approx(0.5)
    assert h1["p_holm"] == pytest.approx(3 * 2 / 1024) and h1["decision"] == "supported"
    assert (h2["only_a"], h2["only_b"]) == (3, 3) and h2["p_raw"] == 1.0 and h2["p_holm"] == 1.0
    assert h2["decision"] == "not_significant"
    assert (h3["only_a"], h3["only_b"]) == (0, 7) and h3["p_raw"] == pytest.approx(2 / 128)
    assert h3["p_holm"] == pytest.approx(2 * 2 / 128) and h3["decision"] == "opposite" and h3["diff"] == pytest.approx(-0.35)
    for h in hyp:
        assert h["ci_lo"] <= h["diff"] <= h["ci_hi"]
    rep = (out / "main_report.md").read_text(encoding="utf-8")
    assert "DRY RUN — NOT RESULTS" in rep and "OPPOSITE RESULT" in rep and "EXPLORATORY" in rep
    assert "skipped_lifted" in rep and "40/40 complete" in rep
    assert sorted(p.name for p in out.glob("fig_*.png")) == ["fig_delay.png", "fig_hypotheses.png", "fig_noise.png", "fig_push.png"]
    assert all(p.stat().st_size > 5000 for p in out.glob("fig_*.png"))
    with open(out / "main_cells.csv", newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 40


def test_analyze_main_push_accounting_and_best_policy(tmp_path):
    am = _load_script("analyze_main")
    seeds = list(range(5000, 5020))
    _synthetic_main(tmp_path, seeds)
    data = am.load_all(tmp_path, seeds)
    acc = {r["cell"]: r for r in am.push_accounting(data)}
    assert len(acc) == 12
    r = acc["k20__push0.04"]
    assert (r["n"], r["pushed"], r["skipped_lifted"], r["not_reached"], r["truncated"]) == (20, 18, 1, 1, 2)
    assert (r["truncated_table"], r["truncated_arm"]) == (1, 1)
    best = {b["condition"]: b for b in am.best_policies(am.cell_table(data))}
    assert len(best) == 10
    # 设计格：push0.06 里 k20 成功 15/20（k100 5/20），k50 / te 是随机的 ~50%：k20 一定在最好之列（可能并列）
    assert "k20" in best["push0.06"]["best"] and best["push0.06"]["rate"] >= 0.75
    for b in best.values():
        assert b["rate"] == max(t["success_rate"] for t in am.cell_table(data) if t["condition"] == b["condition"])


def test_analyze_main_flags_incomplete_cells_and_skips_tests(tmp_path):
    am = _load_script("analyze_main")
    seeds = list(range(5000, 5020))
    _synthetic_main(tmp_path, seeds)
    # 让 H1 里的一格少 3 回合
    p = tmp_path / "k20__push0.06" / "episodes.csv"
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[:-3]) + "\n", encoding="utf-8")
    data = am.load_all(tmp_path, seeds)
    assert not data["k20__push0.06"]["complete"] and data["k20__push0.06"]["issue"]
    hyp = am.run_hypotheses(data)
    assert hyp[0]["available"] is False and hyp[1]["available"] and hyp[2]["available"]
    assert all("p_holm" in h and h["p_holm"] != h["p_holm"] for h in hyp[1:])    # NaN：三个假设没都能检验就不做 Holm
    # 缺格
    (tmp_path / "te__none" / "episodes.csv").unlink()
    assert am.load_cell(tmp_path, "te__none", seeds)["complete"] is False


def test_analyze_main_requires_identical_seed_sets(tmp_path):
    am = _load_script("analyze_main")
    seeds = list(range(5000, 5020))
    _synthetic_main(tmp_path, seeds)
    c = am.load_cell(tmp_path, "k20__none", seeds[:19] + [9999])
    assert not c["complete"]
