"""T9：DP 正式网格、配置生成、run_main_dp 辅助、analyze_main_dp / analyze_interaction 端到端（合成的 episodes.csv，不跑仿真）、
差距变化的手算例子。"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from chunkre import main_dp as md
from chunkre.config import load_config, parse_seeds
from chunkre.stats import gap_change, mcnemar_exact

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name):
    for p in (ROOT / "scripts",):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- 网格（与 PREREGISTRATION_DP.md 一致） ----------
def test_grid_is_3x4_unique():
    cs = md.cells()
    assert len(cs) == 12 and len({c.name for c in cs}) == 12
    assert {c.policy for c in cs} == {"k4", "k8", "k15"} and {c.mode for c in cs} == {"fixed"}
    assert [c.k for c in cs[:3]] == [4, 8, 15]
    by_kind = {}
    for c in cs:
        by_kind.setdefault(c.kind, set()).add(c.strength)
    assert by_kind == {"none": {0.0}, "push": {60.0}, "action_delay": {2.0, 4.0}}   # 没有噪声
    assert md.condition_tags() == ["none", "push60", "delay2", "delay4"]
    assert [c.name for c in cs][:3] == ["k4__none", "k8__none", "k15__none"]


def test_main_dp_cells_are_dp_pilot_cells():
    """正式格子名必须都是 DP 试跑格子名（同一份分析代码才能读试跑数据做演示运行）。"""
    from chunkre.pilot_dp import cells as pilot_cells
    pc = {c.name: c for c in pilot_cells()}
    for c in md.cells():
        assert c.name in pc and pc[c.name] == c


def test_hypotheses_match_preregistration():
    h = {x[0]: x for x in md.HYPOTHESES}
    assert len(h) == 2
    assert h["H4"] == ("H4", "push60", "k4", "k15", "H1")
    assert h["H5"] == ("H5", "delay4", "k15", "k8", "H3")
    names = {c.name for c in md.cells()}
    for _, tag, A, B, _ in md.HYPOTHESES:
        assert f"{A}__{tag}" in names and f"{B}__{tag}" in names


def test_preregistration_file_mentions_the_grid():
    txt = (ROOT / "PREREGISTRATION_DP.md").read_text(encoding="utf-8")
    for s in ("lerobot/diffusion_pusht", "200000–200199", "B = 50", "H4", "H5", "Holm", "20261007", "k ∈ {4, 8, 15}",
              "[20, 80]", "at most 20 times", "gap change", "中文摘要", "10,000"):
        assert s in txt, s
    assert "H4" in txt.split("## 5.")[1].split("## 6.")[0]


def test_main_dp_seeds_are_final_range_and_blocked_without_dp_prereg(tmp_path):
    from chunkre.seed_guard import check_seeds
    s = parse_seeds(md.MAIN_DP_SEEDS)
    assert s[0] == 200000 and s[-1] == 200199 and len(s) == md.MAIN_DP_N == 200
    assert md.MAIN_DP_BATCH == 50 and md.MAIN_DP_N % md.MAIN_DP_BATCH == 0
    with pytest.raises(RuntimeError, match="PREREGISTRATION_DP"):
        check_seeds(s, root=tmp_path, model="dp")


def test_generated_configs_load_and_match_cell(tmp_path):
    paths = md.write_configs(tmp_path)
    assert len(paths) == 12
    for c, p in zip(md.cells(), paths):
        cfg = load_config(p)
        assert cfg.model == "dp" and cfg.mode == "fixed" and cfg.k == c.k
        assert cfg.seeds[0] == 200000 and len(cfg.seeds) == 200 and cfg.batch_size == 50
        text = p.read_text(encoding="utf-8")
        assert "T9 DP 正式评估格子" in text and "main_dp_tools.py gen" in text
        if c.kind == "push":
            assert "kind: push" in text and "strength: 60" in text and "t_lo: 20" in text and "t_hi: 80" in text
        elif c.kind == "action_delay":
            assert "kind: action_delay" in text and f"strength: {c.strength:g}" in text
        else:
            assert "perturb" not in text


def test_committed_main_dp_configs_match_generator(tmp_path):
    """configs/main_dp/ 若已生成，必须与代码一致（run_main_dp.sh 每次都会重新生成，这里防手改）。"""
    d = ROOT / "configs" / "main_dp"
    if not d.exists():
        pytest.skip("configs/main_dp 尚未生成")
    md.write_configs(tmp_path)
    for p in tmp_path.glob("*.yaml"):
        assert (d / p.name).read_bytes() == p.read_bytes(), p.name


def test_parse_selection():
    names = [c.name for c in md.cells()]
    assert md.parse_selection(None, names) == names and md.parse_selection("all", names) == names
    assert md.parse_selection("1-6", names) + md.parse_selection("7-12", names) == names
    assert md.parse_selection("k4__none,k15__delay4", names) == ["k4__none", "k15__delay4"]
    with pytest.raises(ValueError):
        md.parse_selection("1-13", names)
    with pytest.raises(ValueError):
        md.parse_selection("k2__none", names)


def test_run_main_dp_script_uses_dp_guard_and_resumes():
    sh = (ROOT / "scripts" / "run_main_dp.sh").read_text(encoding="utf-8")
    assert "\r" not in sh and sh.startswith("#!/bin/bash")
    assert "model='dp'" in sh                      # 开跑前就按 DP 登记文件检查种子
    assert "$MT done" in sh           # 可续跑：已完成的格子跳过
    assert "$MT eta" in sh            # 开头打印预计耗时
    assert "analyze_main_dp.py" in sh and "analyze_interaction.py" in sh   # 跑完自动分析


def test_main_dp_tools_eta_from_committed_pilot():
    mt = _load_script("main_dp_tools")
    bs = mt.pilot_batch_seconds()
    assert set(bs) == {c.name for c in md.cells()}       # 试跑数据已入库，12 格都有
    t = mt.eta_pilot_based(md.cells(), 200, 50)
    assert t is not None and 3600 < t < 6 * 3600         # 约几个小时，数量级检查
    assert mt.eta_pilot_based(md.cells()[:3], 200, 50) < t


# ---------- 差距变化：手算例子 ----------
def test_gap_change_hand_example():
    """8 个种子（0/1 成败）：
    短 k 扰动 s_c = [1,1,1,0,0,0,0,0]（3/8）  长 k 扰动 l_c = [1,0,0,0,1,1,1,1]（5/8）
    短 k 无扰 s_0 = [1,1,0,0,0,0,0,0]（2/8）  长 k 无扰 l_0 = [1,1,1,1,1,1,0,0]（6/8）
    扰动下差 s_c-l_c = [0,1,1,0,-1,-1,-1,-1] -> 平均 -2/8 = -0.25（也等于 3/8 - 5/8）
    无扰差   s_0-l_0 = [0,0,-1,-1,-1,-1,0,0] -> 平均 -4/8 = -0.5
    差距变化逐种子 = [0,1,2,1,0,0,-1,-1]，平均 2/8 = 0.25 = (-0.25) - (-0.5)。"""
    sc = [1, 1, 1, 0, 0, 0, 0, 0]
    lc = [1, 0, 0, 0, 1, 1, 1, 1]
    s0 = [1, 1, 0, 0, 0, 0, 0, 0]
    l0 = [1, 1, 1, 1, 1, 1, 0, 0]
    g = gap_change(sc, lc, s0, l0)
    assert g["n"] == 8
    assert g["rate_short_pert"] == 3 / 8 and g["rate_long_pert"] == 5 / 8
    assert g["rate_short_none"] == 2 / 8 and g["rate_long_none"] == 6 / 8
    assert g["gap_pert"] == pytest.approx(-0.25) and g["gap_none"] == pytest.approx(-0.5)
    assert g["gap_change"] == pytest.approx(0.25)
    assert g["gap_change"] == pytest.approx(g["gap_pert"] - g["gap_none"])
    d = np.array([0, 1, 2, 1, 0, 0, -1, -1])
    assert g["gap_change"] == pytest.approx(d.mean())
    # bootstrap 区间：包含点估计、落在逐种子值的范围内，同样输入结果完全相同（固定随机种子）
    assert -1 <= g["ci_lo"] <= g["gap_change"] <= g["ci_hi"] <= 2
    assert gap_change(sc, lc, s0, l0) == g
    # 手算可核对的另一个结论：把无扰动的四个序列换成与扰动下相同 -> 差距变化恒为 0，区间退化为 0
    z = gap_change(sc, lc, sc, lc)
    assert z["gap_change"] == 0 and z["ci_lo"] == 0 and z["ci_hi"] == 0


def test_gap_change_sign_and_validation():
    n = 10
    ones, zeros = [1] * n, [0] * n
    # 短 k 在扰动下全成功、长 k 全失败；无扰动下两者都全成功 -> 差距变化 = +1，区间退化为 1
    g = gap_change(ones, zeros, ones, ones)
    assert g["gap_change"] == 1.0 and (g["ci_lo"], g["ci_hi"]) == (1.0, 1.0)
    # 反过来：长 k 吃亏 -> 取反
    assert gap_change(zeros, ones, ones, ones)["gap_change"] == -1.0
    with pytest.raises(ValueError):
        gap_change([1, 0], [1], [1, 0], [1, 0])
    with pytest.raises(ValueError):
        gap_change([2, 0], [1, 0], [1, 0], [1, 0])
    with pytest.raises(ValueError):
        gap_change([], [], [], [])


# ---------- analyze_main_dp ----------
def _write_cell(root: Path, name: str, seeds, success, push=False, delay=False):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    n = len(seeds)
    with open(d / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "success", "n_policy_calls", "path_length", "jerk", "smoothness",
                                          "max_reward", "perturb_kind", "push_status", "push_clip", "push_redraws"])
        w.writeheader()
        for i, s in enumerate(seeds):
            status = ""
            if push:
                status = "pushed" if i >= 2 else ["skipped_overlap", "not_reached"][i]
            w.writerow(dict(seed=s, success=int(success[i]), n_policy_calls=20 + i % 3, path_length=1000.0 + i,
                            jerk=4.0, smoothness=5.0, max_reward=(1.0 if success[i] else 0.5),
                            perturb_kind=("push" if push else ""), push_status=status,
                            push_clip=("wall" if push and i == 3 else ""),
                            push_redraws=((i % 3) if push and status == "pushed" else "")))


def _synthetic_dp(root: Path, seeds, rng_seed=0):
    """12 格；H4、H5 涉及的 4 格按手算设计（n=20）：
    H4 (k4 vs k15 @ push60)：only_a=0, only_b=9（k15 比 k4 多成功 9 个种子）-> p = 2/512，方向相反；
    H5 (k15 vs k8 @ delay4)：only_a=6, only_b=1 -> n=7, P(X<=1)=(1+7)/128，p = 16/128 = 0.125。
    Holm（族 = 2）：排序 2/512, 0.125 -> 2·(2/512) = 0.0078125, 0.125 -> H4 opposite，H5 not_significant。"""
    rng = np.random.default_rng(rng_seed)
    n = len(seeds)
    z = np.zeros(n, dtype=int)
    k4, k15 = z.copy(), z.copy()
    k15[:9] = 1                      # 只有 k15 成功的 9 个种子
    h5_a, h5_b = z.copy(), z.copy()
    h5_a[:7] = 1                     # k15 @ delay4：前 7 个成功
    h5_b[6] = 1                      # k8 @ delay4：第 7 个成功 -> only_a = 6，only_b = 0 ... 再加 only_b
    h5_b[10] = 1                     # k8 额外成功 1 个 k15 没成功的 -> only_a=6, only_b=1, both=1
    spec = {"k4__push60": k4, "k15__push60": k15, "k15__delay4": h5_a, "k8__delay4": h5_b}
    for c in md.cells():
        succ = spec.get(c.name, (rng.uniform(size=n) < 0.5).astype(int))
        _write_cell(root, c.name, seeds, succ, push=(c.kind == "push"))


def test_analyze_main_dp_end_to_end_hand_designed(tmp_path):
    pytest.importorskip("matplotlib")
    am = _load_script("analyze_main_dp")
    seeds = list(range(5000, 5020))
    _synthetic_dp(tmp_path / "data", seeds)
    out = tmp_path / "out"
    am.main(["--dir", str(tmp_path / "data"), "--out", str(out), "--seeds", "5000-5019"])
    h4, h5 = json.loads((out / "hypotheses_dp.json").read_text(encoding="utf-8"))
    assert (h4["id"], h4["A"], h4["B"], h4["condition"]) == ("H4", "k4", "k15", "push60")
    assert (h4["only_a"], h4["only_b"], h4["both"], h4["neither"]) == (0, 9, 0, 11)
    assert h4["p_raw"] == pytest.approx(2 / 512) == pytest.approx(mcnemar_exact(0, 9))
    assert h4["diff"] == pytest.approx(-0.45)
    assert h4["p_holm"] == pytest.approx(2 * 2 / 512) and h4["decision"] == "opposite"
    assert (h5["id"], h5["A"], h5["B"], h5["condition"]) == ("H5", "k15", "k8", "delay4")
    assert (h5["only_a"], h5["only_b"], h5["both"]) == (6, 1, 1)
    assert h5["p_raw"] == pytest.approx(16 / 128) and h5["p_holm"] == pytest.approx(16 / 128)
    assert h5["decision"] == "not_significant" and h5["diff"] == pytest.approx(0.25)
    for h in (h4, h5):
        assert h["ci_lo"] <= h["diff"] <= h["ci_hi"]
    rep = (out / "main_dp_report.md").read_text(encoding="utf-8")
    assert "DRY RUN — NOT RESULTS" in rep and "OPPOSITE RESULT" in rep and "EXPLORATORY" in rep
    assert "12/12 complete" in rep and "skipped_overlap" in rep and "mean maximum coverage" in rep
    assert "Holm correction over H4–H5 only" in rep
    assert sorted(p.name for p in out.glob("fig_*.png")) == ["fig_hypotheses_dp.png", "fig_success_dp.png"]
    assert all(p.stat().st_size > 5000 for p in out.glob("fig_*.png"))
    with open(out / "main_dp_cells.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12 and "max_reward_mean" in rows[0]


def test_analyze_main_dp_push_accounting_and_best(tmp_path):
    am = _load_script("analyze_main_dp")
    seeds = list(range(5000, 5020))
    _synthetic_dp(tmp_path, seeds)
    data = am.load_all(tmp_path, seeds)
    acc = {r["cell"]: r for r in am.push_accounting(data)}
    assert set(acc) == {"k4__push60", "k8__push60", "k15__push60"}
    r = acc["k4__push60"]
    assert (r["n"], r["pushed"], r["skipped_overlap"], r["not_reached"], r["truncated_wall"]) == (20, 18, 1, 1, 1)
    # push_redraws 只对“推了”的 18 个回合：i=2..19 的 i%3 -> 取值 0/1/2
    assert r["max_redraws"] == 2 and r["redrawn"] == sum(1 for i in range(2, 20) if i % 3 > 0)
    table = am.cell_table(data)
    best = {b["condition"]: b for b in am.best_policies(table)}
    assert len(best) == 4
    # 设计格：push60 里 k15 成功 9/20、k4 0/20，k8 是随机的：最好的成功率一定 >= 0.45，且等于该条件下的最大值
    assert best["push60"]["rate"] >= 0.45 and best["push60"]["rate"] == max(
        t["success_rate"] for t in table if t["condition"] == "push60")


def test_analyze_main_dp_flags_incomplete_cells(tmp_path):
    am = _load_script("analyze_main_dp")
    seeds = list(range(5000, 5020))
    _synthetic_dp(tmp_path, seeds)
    p = tmp_path / "k4__push60" / "episodes.csv"
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[:-3]) + "\n", encoding="utf-8")
    hyp = am.run_hypotheses(am.load_all(tmp_path, seeds))
    assert hyp[0]["available"] is False and hyp[1]["available"]
    assert hyp[1]["p_holm"] != hyp[1]["p_holm"]      # NaN：两个假设没都能检验就不做 Holm
    (tmp_path / "k8__none" / "episodes.csv").unlink()
    assert am.load_cell(tmp_path, "k8__none", seeds)["complete"] is False


# ---------- analyze_interaction ----------
def _write_pair(root: Path, short, long, tag, sc, lc, s0, l0, seeds):
    _write_cell(root, f"{short}__{tag}", seeds, sc)
    _write_cell(root, f"{long}__{tag}", seeds, lc)
    _write_cell(root, f"{short}__none", seeds, s0)
    _write_cell(root, f"{long}__none", seeds, l0)


def test_analyze_interaction_end_to_end(tmp_path):
    pytest.importorskip("matplotlib")
    ai = _load_script("analyze_interaction")
    seeds8 = list(range(5000, 5008))
    sc = [1, 1, 1, 0, 0, 0, 0, 0]
    lc = [1, 0, 0, 0, 1, 1, 1, 1]
    s0 = [1, 1, 0, 0, 0, 0, 0, 0]
    l0 = [1, 1, 1, 1, 1, 1, 0, 0]
    # DP：push60 用手算例子；delay4 全部相同 -> 差距变化 0
    dp, act = tmp_path / "dp", tmp_path / "act"
    _write_pair(dp, "k4", "k15", "push60", sc, lc, s0, l0, seeds8)
    for pref, arr in (("k4", sc), ("k15", lc)):
        _write_cell(dp, f"{pref}__delay4", seeds8, arr)
    # ACT：三个条件，k20 对 k100；push0.06 同手算例子；noise16 少一格（缺格要报告）；delay10 全 0
    _write_pair(act, "k20", "k100", "push0.06", sc, lc, s0, l0, seeds8)
    _write_cell(act, "k20__noise16", seeds8, sc)           # 缺 k100__noise16
    _write_pair(act, "k20", "k100", "delay10", [0] * 8, [0] * 8, s0, l0, seeds8)
    out = tmp_path / "out"
    ai.main(["--act-dir", str(act), "--dp-dir", str(dp), "--act-seeds", "5000-5007", "--dp-seeds", "5000-5007",
             "--out", str(out)])
    res = json.loads((out / "interaction.json").read_text(encoding="utf-8"))
    dp_push, dp_delay = res["DP"]
    assert (dp_push["short"], dp_push["long"], dp_push["condition"]) == ("k4", "k15", "push60")
    assert dp_push["gap_change"] == pytest.approx(0.25) and dp_push["gap_pert"] == pytest.approx(-0.25)
    assert dp_push["gap_none"] == pytest.approx(-0.5) and dp_push["ci_lo"] <= 0.25 <= dp_push["ci_hi"]
    assert dp_delay["condition"] == "delay4" and dp_delay["available"]
    # delay4 的扰动格用 sc/lc，无扰动格用 s0/l0（上面 push60 写入的）：差距变化同样是 0.25
    assert dp_delay["gap_change"] == pytest.approx(0.25)
    a_push, a_noise, a_delay = res["ACT"]
    assert (a_push["short"], a_push["long"]) == ("k20", "k100") and a_push["gap_change"] == pytest.approx(0.25)
    assert a_noise["available"] is False and "k100__noise16" in a_noise["reason"]
    # delay10：两者在扰动下都是 0 -> 扰动下差 0，无扰动差 -0.5 -> 差距变化 +0.5
    assert a_delay["gap_change"] == pytest.approx(0.5) and a_delay["gap_pert"] == 0
    rep = (out / "interaction_report.md").read_text(encoding="utf-8")
    assert "EXPLORATORY" in rep and "no significance test" in rep and "post hoc" in rep
    assert rep.count("DRY RUN — NOT RESULTS") == 2         # 种子 <200000：两个模型都是演示运行
    assert "not computed" in rep
    assert (out / "fig_interaction.png").stat().st_size > 5000
    with open(out / "interaction.csv", newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 5


def test_interaction_comparisons_match_preregistration():
    ai = _load_script("analyze_interaction")
    assert ai.MODELS["ACT"]["short"] == "k20" and ai.MODELS["ACT"]["long"] == "k100"
    assert ai.MODELS["ACT"]["conds"] == ["push0.06", "noise16", "delay10"] and ai.MODELS["ACT"]["post_hoc"]
    assert ai.MODELS["DP"]["short"] == "k4" and ai.MODELS["DP"]["long"] == "k15"
    assert ai.MODELS["DP"]["conds"] == ["push60", "delay4"] and not ai.MODELS["DP"]["post_hoc"]
    from chunkre import main_grid as mg
    act_names = {c.name for c in mg.cells()}
    for c in ai.MODELS["ACT"]["conds"] + ["none"]:
        assert f"k20__{c}" in act_names and f"k100__{c}" in act_names
    dp_names = {c.name for c in md.cells()}
    for c in ai.MODELS["DP"]["conds"] + ["none"]:
        assert f"k4__{c}" in dp_names and f"k15__{c}" in dp_names
