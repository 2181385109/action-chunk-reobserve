"""T10b：自训练 ACT 的网格、配置、登记文件一致性、run_main_own 辅助、analyze_main_own 端到端（合成的 episodes.csv，不跑仿真）。"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from chunkre import main_grid as mg
from chunkre import main_own as mo
from chunkre.config import load_config, parse_seeds
from chunkre.stats import holm_adjust, mcnemar_exact

ROOT = Path(__file__).resolve().parent.parent
SHA_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def _load_script(name):
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- 网格（与 PREREGISTRATION_OWN.md 一致） ----------
def test_grid_is_3x3_and_subset_of_act_main_grid():
    cs = mo.cells()
    assert len(cs) == 9 and len({c.name for c in cs}) == 9
    assert [c.k for c in cs[:3]] == [20, 50, 100] and {c.mode for c in cs} == {"fixed"}
    assert mo.condition_tags() == ["none", "push0.06", "delay10"]
    by_kind = {}
    for c in cs:
        by_kind.setdefault(c.kind, set()).add(c.strength)
    assert by_kind == {"none": {0.0}, "push": {0.06}, "action_delay": {10.0}}   # 没有噪声、没有时间集成
    main = {c.name: c for c in mg.cells()}
    for c in cs:   # 格子（含扰动设置）与预训练模型的 ACT 正式网格完全相同 -> 同一份分析代码可读 results/main/
        assert main[c.name] == c


def test_hypotheses_and_gap_match_preregistration():
    assert mo.HYPOTHESES == [("H1'", "push0.06", "k20", "k100", "H1"), ("H3'", "delay10", "k100", "k50", "H3")]
    names = {c.name for c in mo.cells()}
    for _, tag, A, B, _ in mo.HYPOTHESES:
        assert f"{A}__{tag}" in names and f"{B}__{tag}" in names
    assert mo.GAP == dict(short="k20", long="k100", conds=["push0.06", "delay10"])


def test_seeds_baseline_and_constants():
    s = parse_seeds(mo.OWN_SEEDS)
    assert (s[0], s[-1], len(s)) == (200000, 200199, 200) and mo.OWN_N == 200
    b = parse_seeds(mo.BASELINE_SEEDS)
    assert (b[0], b[-1], len(b)) == (1000, 1499, 500) and mo.BASELINE_N == 500
    assert mo.OWN_BATCH == 10 and mo.BASELINE_K == 100 and mo.WEAK_THRESHOLD == 0.40 and mo.OFFICIAL_SUCCESS == 0.83
    assert mo.CHECKPOINT.endswith("/checkpoints/100000/pretrained_model")   # 最终检查点，写死


def test_preregistration_file_matches_code():
    txt = (ROOT / "PREREGISTRATION_OWN.md").read_text(encoding="utf-8")
    for s in (mo.CHECKPOINT, mo.CHECKPOINT_SHA256, "200000–200199", "1000–1499", "B = 10", "H1'", "H3'", "Holm", "20261007",
              "k ∈ {20, 50, 100}", "[20, 100]", "0.06 m", "10 steps", "below 40%", "gap change", "agreement", "中文摘要",
              "before any evaluation result", "seed 1000", "10,000", "82.6%", "83.0%"):
        assert s in txt, s
    assert "H1'" in txt.split("## 5.")[1].split("## 6.")[0]


# ---------- 配置 ----------
def test_generated_configs_load_and_match_act_main_configs(tmp_path):
    paths = mo.write_configs(tmp_path)
    assert len(paths) == 10
    for c, p in zip(mo.cells(), paths):
        cfg = load_config(p)
        assert cfg.model == "act" and cfg.mode == "fixed" and cfg.k == c.k and cfg.batch_size == 10
        assert cfg.policy_path == mo.CHECKPOINT
        assert cfg.seeds[0] == 200000 and len(cfg.seeds) == 200
        # 扰动设置与预训练模型的 ACT 正式配置逐项相同（只多一行 policy_path，注释头不同）
        mp = ROOT / "configs" / "main" / f"{c.name}.yaml"
        if mp.exists():
            own = [ln for ln in p.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
            main = [ln for ln in mp.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
            assert own == main + [f"policy_path: {mo.CHECKPOINT}"], c.name
    base = load_config(paths[-1])
    assert base.name == mo.BASELINE_NAME and base.k == 100 and base.mode == "fixed" and base.perturb_active is False
    assert (base.seeds[0], base.seeds[-1], len(base.seeds), base.batch_size) == (1000, 1499, 500, 10)
    assert base.policy_path == mo.CHECKPOINT


def test_committed_main_own_configs_match_generator(tmp_path):
    d = ROOT / "configs" / "main_own"
    if not d.exists():
        pytest.skip("configs/main_own 尚未生成")
    mo.write_configs(tmp_path)
    for p in tmp_path.glob("*.yaml"):
        assert (d / p.name).read_bytes() == p.read_bytes(), p.name


def test_parse_selection():
    names = [c.name for c in mo.cells()]
    assert mo.parse_selection(None, names) == names and mo.parse_selection("all", names) == names
    assert mo.parse_selection("1-5", names) + mo.parse_selection("6-9", names) == names
    assert mo.parse_selection("k20__none,k100__delay10", names) == ["k20__none", "k100__delay10"]
    with pytest.raises(ValueError):
        mo.parse_selection("1-10", names)
    with pytest.raises(ValueError):
        mo.parse_selection("te__none", names)


# ---------- run_main_own.sh / 辅助 ----------
def test_run_main_own_script_guards_and_resumes():
    sh = (ROOT / "scripts" / "run_main_own.sh").read_text(encoding="utf-8")
    assert "\r" not in sh and sh.startswith("#!/bin/bash")
    assert "$MT guard" in sh and "$MT verify-ckpt" in sh      # 登记保护（含基础水平测试）+ 检查点哈希核对，先于一切
    assert sh.index("$MT guard") < sh.index("eval.py") and sh.index("$MT verify-ckpt") < sh.index("eval.py")
    assert "$MT done" in sh and "$MT eta" in sh               # 可续跑 / 开头打印预计耗时
    assert sh.index("own_baseline.yaml") < sh.index('configs/main_own/$c.yaml')   # 先基础水平测试，再 9 格
    assert "analyze_main_own.py" in sh                        # 跑完自动分析
    assert "checkpoints/last" not in sh                       # 不用 last，写死的第 100000 步在 main_own.py


def test_main_own_tools_eta_and_checkpoint_helpers(tmp_path):
    mt = _load_script("main_own_tools")
    t = mt.eta_measured(mo.cells(), True)
    assert t is not None and 2 * 3600 < t < 6 * 3600          # 约 3.6 小时，数量级检查
    assert mt.eta_measured(mo.cells(), False) < t
    assert mt.eta_measured(mo.cells()[:3], False) < mt.eta_measured(mo.cells(), False)
    assert mt.checkpoint_sha256(str(tmp_path)) is None         # 没有 model.safetensors
    (tmp_path / "model.safetensors").write_bytes(b"abc")
    assert mt.checkpoint_sha256(str(tmp_path)) == SHA_ABC
    assert mt.checkpoint_sha256(str(tmp_path)) != mo.CHECKPOINT_SHA256


def test_local_policy_hashes(tmp_path):
    from chunkre.runner import local_policy_hashes
    (tmp_path / "model.safetensors").write_bytes(b"abc")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    h = local_policy_hashes(tmp_path)
    assert set(h) == {"model.safetensors", "config.json"}      # 没有 train_config.json 就不记
    assert h["model.safetensors"] == SHA_ABC


# ---------- analyze_main_own（合成数据） ----------
def _write_cell(root: Path, name: str, seeds, success, push=False, registered=True, wrong_hash=False):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "success", "n_policy_calls", "path_length", "jerk", "smoothness",
                                          "perturb_kind", "push_status", "push_clip"])
        w.writeheader()
        for i, s in enumerate(seeds):
            w.writerow(dict(seed=s, success=int(success[i]), n_policy_calls=5 + i % 3, path_length=8.0 + i * 0.01, jerk=0.02,
                            smoothness=0.03, perturb_kind=("push" if push else ""),
                            push_status=("pushed" if push and i % 7 else "skipped_lifted" if push else ""),
                            push_clip=("table" if push and i % 11 == 0 else "")))
    if registered:
        sha = "0" * 64 if wrong_hash else mo.CHECKPOINT_SHA256
        pol = dict(repo=mo.CHECKPOINT, local_checkpoint_sha256={"model.safetensors": sha})
    else:
        pol = dict(repo="lerobot/act_aloha_sim_transfer_cube_human")
    (d / "config.json").write_text(json.dumps(dict(policy=pol)), encoding="utf-8")


RATES = {"none": (0.6, 0.7, 0.75), "push0.06": (0.5, 0.4, 0.2), "delay10": (0.0, 0.2, 0.45)}   # k20, k50, k100


def _synthetic(root: Path, base: Path, seeds, bseeds, rates=RATES, base_rate=0.8, seed0=0, **kw):
    rng = np.random.default_rng(seed0)
    for tag, rs in rates.items():
        for pol, r in zip(("k20", "k50", "k100"), rs):
            _write_cell(root, f"{pol}__{tag}", seeds, (rng.random(len(seeds)) < r).astype(int), push=tag.startswith("push"), **kw)
    _write_cell(base.parent, base.name, bseeds, (rng.random(len(bseeds)) < base_rate).astype(int), **kw)


def test_analyze_main_own_end_to_end(tmp_path):
    am = _load_script("analyze_main_own")
    seeds, bseeds = parse_seeds(mo.OWN_SEEDS), parse_seeds(mo.BASELINE_SEEDS)
    own, base = tmp_path / "main_own", tmp_path / "own_baseline"
    pre, pre_base = tmp_path / "pre", tmp_path / "prebase"
    _synthetic(own, base, seeds, bseeds, seed0=1)
    _synthetic(pre, pre_base, seeds, bseeds, seed0=2, registered=False)
    out = tmp_path / "out"
    am.main(["--dir", str(own), "--baseline-dir", str(base), "--out", str(out), "--pretrained-dir", str(pre),
             "--pretrained-baseline-dir", str(pre_base)])
    rep = (out / "main_own_report.md").read_text(encoding="utf-8")
    assert "DRY RUN" not in rep and "all 10 result directories record the registered checkpoint" in rep
    hyp = {h["id"]: h for h in json.loads((out / "hypotheses_own.json").read_text(encoding="utf-8"))}
    assert set(hyp) == {"H1'", "H3'"} and all(h["available"] for h in hyp.values())
    # 手算核对：McNemar p 值 = 不一致对的精确二项；Holm 族 = 这两个
    h1 = hyp["H1'"]
    assert h1["p_raw"] == pytest.approx(mcnemar_exact(h1["only_a"], h1["only_b"]))
    adj = holm_adjust([hyp["H1'"]["p_raw"], hyp["H3'"]["p_raw"]])
    assert hyp["H1'"]["p_holm"] == pytest.approx(adj[0]) and hyp["H3'"]["p_holm"] == pytest.approx(adj[1])
    assert h1["diff"] > 0 and h1["decision"] == "supported" and hyp["H3'"]["decision"] == "supported"
    assert h1["rate_A"] == pytest.approx((h1["only_a"] + h1["both"]) / 200)
    b = json.loads((out / "baseline_own.json").read_text(encoding="utf-8"))
    assert b["n"] == 500 and b["weak"] is False and b["official"] == 0.83 and "repro_rate" in b
    conc = json.loads((out / "concordance_own.json").read_text(encoding="utf-8"))
    assert len(conc) == 10 and all(c["available"] for c in conc)
    for c in conc:
        assert c["both"] + c["only_own"] + c["only_pretrained"] + c["neither"] == c["n"]
        assert c["agreement"] == pytest.approx((c["both"] + c["neither"]) / c["n"])
    gaps = json.loads((out / "interaction_own.json").read_text(encoding="utf-8"))
    assert {(g["model"], g["condition"]) for g in gaps} == {(m, c) for m in ("own", "pretrained") for c in ("push0.06", "delay10")}
    for f in ("main_own_cells.csv", "fig_hypotheses_own.png", "fig_success_own.png", "fig_interaction_own.png",
              "concordance_own.csv", "interaction_own.csv"):
        assert (out / f).exists(), f


def test_analyze_main_own_marks_dry_run_for_wrong_checkpoint(tmp_path):
    """没有登记检查点记录的数据（如预训练模型的结果、或别的检查点）永远不会被标成正式结果。"""
    am = _load_script("analyze_main_own")
    seeds, bseeds = parse_seeds(mo.OWN_SEEDS), parse_seeds(mo.BASELINE_SEEDS)
    for label, kw in (("hub", dict(registered=False)), ("hash", dict(wrong_hash=True))):
        own, base = tmp_path / label / "main_own", tmp_path / label / "own_baseline"
        _synthetic(own, base, seeds, bseeds, **kw)
        out = tmp_path / label / "out"
        am.main(["--dir", str(own), "--baseline-dir", str(base), "--out", str(out), "--pretrained-dir", str(own),
                 "--pretrained-baseline-dir", str(base)])
        rep = (out / "main_own_report.md").read_text(encoding="utf-8")
        assert rep.startswith("> **DRY RUN — NOT RESULTS.**"), label
        assert "Checkpoint check failed" in rep and "trivially 100%" in rep
    # --dry-run 强制标注（即使检查点记录正确）
    own, base = tmp_path / "ok" / "main_own", tmp_path / "ok" / "own_baseline"
    _synthetic(own, base, seeds, bseeds)
    am.main(["--dir", str(own), "--baseline-dir", str(base), "--out", str(tmp_path / "ok" / "out"), "--pretrained-dir", str(own),
             "--pretrained-baseline-dir", str(base), "--dry-run"])
    assert (tmp_path / "ok" / "out" / "main_own_report.md").read_text(encoding="utf-8").startswith("> **DRY RUN")


def test_analyze_main_own_weak_model_rule_and_incomplete(tmp_path):
    am = _load_script("analyze_main_own")
    seeds, bseeds = parse_seeds(mo.OWN_SEEDS), parse_seeds(mo.BASELINE_SEEDS)
    own, base = tmp_path / "main_own", tmp_path / "own_baseline"
    _synthetic(own, base, seeds, bseeds, base_rate=0.3, seed0=3)
    # 删掉 k50__delay10 的最后 10 行 -> 不完整，H3' 不能检验，不做 Holm
    p = own / "k50__delay10" / "episodes.csv"
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[:-10]) + "\n", encoding="utf-8")
    pre, pre_base = tmp_path / "pre", tmp_path / "prebase"
    _synthetic(pre, pre_base, seeds, bseeds, seed0=4, registered=False)
    out = tmp_path / "out"
    am.main(["--dir", str(own), "--baseline-dir", str(base), "--out", str(out), "--pretrained-dir", str(pre),
             "--pretrained-baseline-dir", str(pre_base)])
    rep = (out / "main_own_report.md").read_text(encoding="utf-8")
    assert "Weak-model rule triggered" in rep and "limited reference value" in rep
    assert "INCOMPLETE" in rep
    hyp = {h["id"]: h for h in json.loads((out / "hypotheses_own.json").read_text(encoding="utf-8"))}
    assert hyp["H1'"]["available"] and not hyp["H3'"]["available"]
    assert hyp["H1'"]["decision"].startswith("not computed")


def test_dryrun_on_committed_pretrained_data_runs(tmp_path):
    """results/main/（预训练模型）对应的 9 格 + results/repro/act 喂给分析脚本（演示运行）：能跑通、标 DRY RUN、数字与 results/main 一致。"""
    am = _load_script("analyze_main_own")
    if not (ROOT / "results" / "main" / "k20__push0.06" / "episodes.csv").exists():
        pytest.skip("results/main 不在")
    am.main(["--dir", "results/main", "--baseline-dir", "results/repro/act", "--out", str(tmp_path), "--dry-run"])
    rep = (tmp_path / "main_own_report.md").read_text(encoding="utf-8")
    assert rep.startswith("> **DRY RUN — NOT RESULTS.**")
    hyp = {h["id"]: h for h in json.loads((tmp_path / "hypotheses_own.json").read_text(encoding="utf-8"))}
    main_h = {h["id"]: h for h in json.loads((ROOT / "results" / "main" / "hypotheses.json").read_text(encoding="utf-8"))}
    assert hyp["H1'"]["diff"] == pytest.approx(main_h["H1"]["diff"]) and hyp["H1'"]["p_raw"] == pytest.approx(main_h["H1"]["p_raw"])
    assert hyp["H3'"]["diff"] == pytest.approx(main_h["H3"]["diff"]) and hyp["H3'"]["p_raw"] == pytest.approx(main_h["H3"]["p_raw"])
    # Holm 族变小（2 而不是 3）：校正后 p 不大于 ACT 里 3 个假设一起校正的
    assert hyp["H1'"]["p_holm"] <= main_h["H1"]["p_holm"] + 1e-15
