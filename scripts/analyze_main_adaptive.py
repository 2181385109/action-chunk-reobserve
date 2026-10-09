"""T11b 自适应规则正式分析：严格按 PREREGISTRATION_ADAPTIVE.md 实现 HA1（推 0.06，McNemar）、HA2（无扰动，非劣检验）、
描述性对比（k20/k50/k100/te/自适应）、换计划统计、推之后第一次差异换计划的步数分布。

用法（WSL 内；run_main_adaptive.sh 在 2 格全部完成后自动调用）：
  python scripts/analyze_main_adaptive.py [--dir results/main_adaptive] [--out results/main_adaptive]
演示运行（用阈值试跑的数据，调参种子 6000–6049，不是结果）：
  python scripts/analyze_main_adaptive.py --dir results/tune_adaptive --ad-policy ad_m10_q90 --control-dir results/tune_adaptive \
      --extra-control-dir results/main --seeds 6000-6049 --out results/main_adaptive_dryrun --dry-run
  （k=100 对照取试跑里同种子的格子，所以 HA1/HA2 是配对的；k20/k50/te 只有 results/main 里的（种子 200000–200199），
   和试跑种子不配对，报告里只列成功率 / 调用次数，不做配对差。）

输出（--out）：main_adaptive_report.md、hypotheses_adaptive.json、main_adaptive_cells.csv、fig_adaptive.png。
DRY RUN 判定：加了 --dry-run，或种子 <200000，或任一自适应格 config.json 里的参数 / 模式与登记不一致 -> 报告开头标“DRY RUN，不是结果”。
所有回合都计入，不剔除；某格回合数 / 种子集合与登记的不一致时，该格标为不完整，涉及它的检验不做。
HA1/HA2、McNemar、bootstrap、“相反结果”的定义见登记 §5（chunkre.stats）；不做 Holm。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_main import SECONDARY, _fmt, _num  # noqa: E402
from chunkre import main_adaptive as ma  # noqa: E402
from chunkre.config import parse_seeds  # noqa: E402
from chunkre.metrics import read_episodes, wilson_ci  # noqa: E402
from chunkre.plotting import yerr_from_ci  # noqa: E402
from chunkre.stats import (ALPHA, BOOT_SEED, N_BOOT, bootstrap_mean_ci, classify, mcnemar_exact,  # noqa: E402
                           pair_counts, paired_bootstrap_diff)

POLICY_LABEL = {"k20": "k=20", "k50": "k=50", "k100": "k=100", "te": "temporal ensemble", "ad": "adaptive"}
NAN = float("nan")


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------- 读数据
def load_cell(cell_dir: Path, name: str, expected: list[int] | None) -> dict:
    """读一格。expected = 预期种子集合（None = 不检查，用它自己的）。success 按种子升序。"""
    p = Path(cell_dir) / name / "episodes.csv"
    if not p.exists():
        return dict(name=name, dir=Path(cell_dir) / name, rows=[], seeds=[], complete=False, issue="missing (no episodes.csv)")
    rows = sorted(read_episodes(p), key=lambda r: int(float(r["seed"])))
    seeds = [int(float(r["seed"])) for r in rows]
    complete = True if expected is None else seeds == sorted(expected)
    issue = "" if complete else f"{len(rows)} episodes, expected {len(expected)} with seeds {min(expected)}–{max(expected)}"
    return dict(name=name, dir=Path(cell_dir) / name, rows=rows, seeds=seeds,
                success=np.array([int(float(r["success"])) for r in rows]), complete=complete, issue=issue)


def find_fixed(policy: str, tag: str, dirs: list[Path], expected: list[int]) -> dict:
    """k20 / k50 / k100 / te 在条件 tag 下的格子：依次在 dirs 里找第一个有数据的目录。
    paired = 种子集合与自适应格完全相同（才能做配对比较）。"""
    for d in dirs:
        c = load_cell(d, f"{policy}__{tag}", None)
        if c["rows"]:
            c["paired"] = c["seeds"] == sorted(expected)
            c["source"] = str(d)
            return c
    return dict(name=f"{policy}__{tag}", rows=[], seeds=[], complete=False, paired=False, issue="missing", source="")


def adaptive_provenance(cell: dict, reg: dict) -> tuple[bool, str]:
    """自适应格 config.json：mode=adaptive 且 m / tau / h_cmp / min_gap 与登记逐项相同、模型是 hub ACT（没有本地检查点）。"""
    p = Path(cell["dir"]) / "config.json"
    if not p.exists():
        return False, "no config.json"
    try:
        fp = json.loads(p.read_text(encoding="utf-8"))["eval_config_fingerprint"]
    except Exception as e:   # noqa: BLE001
        return False, f"unreadable config.json ({e})"
    if fp.get("mode") != "adaptive":
        return False, f"mode = {fp.get('mode')!r}, not adaptive"
    if fp.get("policy_path"):
        return False, "local checkpoint (not the pretrained hub model)"
    got = fp.get("adaptive") or {}
    bad = [f"{k} = {got.get(k)!r} (registered {reg[k]!r})" for k in ma.PARAM_KEYS
           if got.get(k) is None or float(got[k]) != float(reg[k])]
    if bad:
        return False, "parameters differ from the registration: " + "; ".join(bad)
    return True, "registered parameters"


# ---------------------------------------------------------------- 主要假设
def run_hypotheses(ad: dict[str, dict], k100: dict[str, dict]) -> dict:
    res = {}
    for hid, tag in (("HA1", "push0.06"), ("HA2", "none")):
        a, b = ad.get(tag), k100.get(tag)
        r = dict(id=hid, condition=tag, A="adaptive", B="k100")
        if not (a and b and a["rows"] and b["rows"] and a["complete"] and b.get("paired", b["complete"])):
            why = "; ".join(f"{c['name']}: {c.get('issue') or 'seeds not paired with the adaptive cell'}"
                            for c in (a, b) if c and (not c["rows"] or not c["complete"] or not c.get("paired", True)))
            r.update(available=False, reason=why or "missing")
            res[hid] = r
            continue
        x, y = a["success"], b["success"]
        pc = pair_counts(x, y)
        diff, lo, hi = paired_bootstrap_diff(x, y)
        p = mcnemar_exact(pc["only_a"], pc["only_b"])
        r.update(available=True, n=int(x.size), rate_A=float(x.mean()), rate_B=float(y.mean()), diff=diff, ci_lo=lo, ci_hi=hi,
                 p_mcnemar=p, **pc)
        if hid == "HA1":
            r["decision"] = classify(diff, p)         # supported / opposite / not_significant（不做 Holm）
        else:
            r["margin"] = ma.NONINF_MARGIN
            r["holds"] = bool(lo > -ma.NONINF_MARGIN)
            r["decision"] = "holds" if r["holds"] else "not shown"
        res[hid] = r
    ha1, ha2 = res["HA1"], res["HA2"]
    if ha1["available"] and ha2["available"]:
        res["conclusion"] = dict(
            useful=bool(ha1["decision"] == "supported" and ha2["holds"]),
            text=("HA1 supported and HA2 holds: the adaptive rule is useful on this task (as registered)."
                  if ha1["decision"] == "supported" and ha2["holds"] else
                  "HA1 supported, but non-inferiority without perturbation (HA2) is not shown." if ha1["decision"] == "supported" else
                  "HA1 shows an OPPOSITE RESULT (k = 100 better); no usefulness claimed." if ha1["decision"] == "opposite" else
                  "HA1 not supported (not significant); no usefulness claimed whatever HA2 says."))
    else:
        res["conclusion"] = dict(useful=None, text="not computed (a hypothesis is not testable)")
    return res


# ---------------------------------------------------------------- 描述性对比（登记 §6.1）
def descriptive(ad: dict[str, dict], fixed: dict[str, dict[str, dict]]) -> list[dict]:
    """每个条件：k20/k50/k100/te/自适应 的成功率（Wilson）、平均策略调用；与自适应的配对差（有 bootstrap 区间，无 p 值）。"""
    out = []
    for tag, _, _ in ma.CONDITIONS:
        a = ad.get(tag)
        for pol in ma.FIXED_POLICIES + [ma.POLICY]:
            c = a if pol == ma.POLICY else fixed[pol].get(tag)
            row = dict(condition=tag, policy=pol, available=bool(c and c["rows"]))
            if row["available"]:
                s = c["success"]
                lo, hi = wilson_ci(int(s.sum()), len(s))
                calls = [_num(r.get("n_policy_calls")) for r in c["rows"]]
                row.update(n=len(s), n_success=int(s.sum()), rate=float(s.mean()), wilson_lo=lo, wilson_hi=hi,
                           mean_calls=float(np.nanmean(calls)))
                if pol != ma.POLICY and a and a["rows"]:
                    if c.get("paired"):
                        d, dlo, dhi = paired_bootstrap_diff(a["success"], s)
                        row.update(paired=True, diff_vs_adaptive=d, diff_lo=dlo, diff_hi=dhi)
                    else:
                        row.update(paired=False)
            out.append(row)
    return out


# ---------------------------------------------------------------- 换计划统计 / 次要指标
def replan_stats(ad: dict[str, dict]) -> list[dict]:
    out = []
    for tag, _, _ in ma.CONDITIONS:
        c = ad.get(tag)
        if not (c and c["rows"]):
            continue
        d = np.array([_num(r.get("n_replans_diff")) for r in c["rows"]])
        x = np.array([_num(r.get("n_replans_exhaust")) for r in c["rows"]])
        k = np.array([_num(r.get("n_checks")) for r in c["rows"]])
        n = np.array([_num(r.get("n_policy_calls")) for r in c["rows"]])
        hist = {str(i): int((d == i).sum()) for i in range(4)}
        hist["4+"] = int((d >= 4).sum())
        out.append(dict(condition=tag, n=len(d), mean_diff_replans=float(d.mean()), median_diff_replans=float(np.median(d)),
                        max_diff_replans=int(d.max()), frac_with_diff_replan=float((d >= 1).mean()), hist_diff_replans=hist,
                        mean_exhaust_replans=float(x.mean()), mean_checks=float(k.mean()), mean_policy_calls=float(n.mean())))
    return out


def secondary_table(ad: dict[str, dict]) -> list[dict]:
    out = []
    for tag, _, _ in ma.CONDITIONS:
        c = ad.get(tag)
        row = dict(cell=f"ad__{tag}", condition=tag, n=len(c["rows"]) if c else 0)
        if c and c["rows"]:
            for col, _, _ in SECONDARY:
                m, lo, hi, nv = bootstrap_mean_ci([_num(r.get(col)) for r in c["rows"]])
                row.update({f"{col}_mean": m, f"{col}_lo": lo, f"{col}_hi": hi, f"{col}_n": nv})
        out.append(row)
    return out


def push_accounting(c: dict | None) -> dict | None:
    if not (c and c["rows"]):
        return None
    st = [r.get("push_status", "") for r in c["rows"]]
    clip = [r.get("push_clip", "") for r in c["rows"]]
    return dict(n=len(st), pushed=st.count("pushed"), skipped_lifted=st.count("skipped_lifted"), not_reached=st.count("not_reached"),
                zero=st.count("zero"), truncated=sum(1 for x in clip if x), truncated_table=sum(1 for x in clip if "table" in x),
                truncated_arm=sum(1 for x in clip if "arm" in x))


def reaction(c: dict | None) -> dict | None:
    """推 0.06 格：推之后第一次差异触发的换计划距推的步数分布（登记 §6.3）。"""
    if not (c and c["rows"]):
        return None
    try:
        rp = ma.load_replans(c["dir"])
    except Exception as e:   # noqa: BLE001
        return dict(available=False, reason=f"cannot read adaptive/batch_*.npz ({e})")
    if not rp:
        return dict(available=False, reason="no adaptive/batch_*.npz re-plan records")
    r = ma.first_reaction_steps(c["rows"], rp)
    s = np.array(r["steps"], dtype=float)
    out = dict(available=True, n_pushed=r["n_pushed"], n_with_reaction=int(s.size), n_no_reaction=r["n_no_reaction"],
               n_replan_at_push_step=r["n_replan_at_push_step"], steps=[int(x) for x in s])
    if s.size:
        q = np.percentile(s, [25, 50, 75])
        out.update(min=int(s.min()), q1=float(q[0]), median=float(q[1]), q3=float(q[2]), max=int(s.max()),
                   frac_within={str(w): float((s <= w).mean()) for w in (10, 20, 30)})
    return out


# ---------------------------------------------------------------- 报告
def _rate(r: dict) -> str:
    return "—" if not r.get("available") else f"{r['n_success']}/{r['n']} = {r['rate']:.3f} [{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}]"


def build_report(ctx: dict) -> str:
    hyp = ctx["hyp"]
    L = []
    if ctx["dry_run"]:
        L += ["> **DRY RUN — NOT RESULTS.** " + ctx["dry_reason"] + " This report only checks that the analysis code runs. "
              "Nothing here is a finding about the adaptive rule.", ""]
        if ctx["unpaired_note"]:
            L += ["> " + ctx["unpaired_note"], ""]
    L += ["# Adaptive re-observation rule — analysis report (T11b, implements PREREGISTRATION_ADAPTIVE.md)", ""]
    reg = ctx["reg"]
    L.append(f"- Registered rule: m = {reg['m']}, tau = {reg['tau']}, h_cmp = {reg['h_cmp']}, min_gap = {reg['min_gap']}, L = 100; "
             "pretrained ACT only.")
    for p in ctx["provenance"]:
        if not p["ok"]:
            L.append(f"- **Parameter / model check failed** for {p['item']}: {p['why']}.")
    if ctx["provenance"] and all(p["ok"] for p in ctx["provenance"]):
        L.append(f"- Parameter check: all {len(ctx['provenance'])} adaptive cells record exactly the registered parameters (config.json).")
    L.append(f"- Cells: expected {ctx['n']} episodes per cell, seeds {ctx['seeds_label']}; control = k = 100 cells of "
             f"`{ctx['control_label']}` (same seeds, paired).")
    L.append(f"- Tests: exact two-sided McNemar on seed-paired outcomes (HA1); paired bootstrap 95% CI ({N_BOOT} resamples, RNG seed "
             f"{BOOT_SEED}); non-inferiority margin {ma.NONINF_MARGIN:.2f} (HA2); alpha = {ALPHA}; **no Holm** (intersection–union).")
    for i in ctx["issues"]:
        L.append(f"- **INCOMPLETE** {i}")
    L.append("")

    L += ["## Primary hypotheses (confirmatory)", ""]
    L += ["| H | condition | n | adaptive | k=100 | diff (adaptive − k=100) | 95% CI (paired bootstrap) | only adaptive / only k=100 | p (McNemar) | decision |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    names = {"supported": "supported", "opposite": "OPPOSITE RESULT", "not_significant": "not significant",
             "holds": "non-inferiority HOLDS (lower limit > −0.05)", "not shown": "non-inferiority NOT SHOWN (lower limit ≤ −0.05)"}
    for hid in ("HA1", "HA2"):
        r = hyp[hid]
        if not r["available"]:
            L.append(f"| {hid} | {r['condition']} | — | — | — | — | — | — | — | not testable: {r['reason']} |")
            continue
        p_txt = f"{r['p_mcnemar']:.4g}" + (" (descriptive only)" if hid == "HA2" else "")
        L.append(f"| {hid} | {r['condition']} | {r['n']} | {r['rate_A']:.3f} | {r['rate_B']:.3f} | {r['diff']:+.3f} | "
                 f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['only_a']} / {r['only_b']} | {p_txt} | {names[r['decision']]} |")
    L += ["", "HA1: success rate of the adaptive rule is higher than that of k = 100 under push 0.06 m (two-sided exact McNemar, alpha = 0.05; "
              "\"OPPOSITE RESULT\" = p < 0.05 with k = 100 better). HA2: with no perturbation the adaptive rule is not inferior to k = 100 by more than "
              f"{100 * ma.NONINF_MARGIN:.0f} percentage points (lower limit of the paired bootstrap 95% CI of adaptive − k=100 is > −{ma.NONINF_MARGIN:.2f}); "
              "McNemar there is descriptive only. With 200 paired seeds the CI half-width is about ±6 points, so a failure of HA2 "
              "means \"not shown at this sample size\", not \"worse\" (PREREGISTRATION_ADAPTIVE.md §5).", "",
          f"**Conclusion (registered rule): {hyp['conclusion']['text']}**", ""]

    L += ["## Descriptive comparison with the fixed policies (exploratory; no tests, not evidence)", "",
          "Success rate (Wilson 95% CI) and mean policy calls per episode. **k = 50 is the cheaper fixed control** (stated in advance in the "
          "registration); the difference \"adaptive − X\" is given with a paired bootstrap 95% CI where the seeds are paired (no p-values).", ""]
    for tag, _, _ in ma.CONDITIONS:
        L += [f"### {tag}", "", "| policy | success | mean policy calls | adaptive − policy (paired bootstrap 95% CI) |", "|---|---|---|---|"]
        for r in [x for x in ctx["desc"] if x["condition"] == tag]:
            lab = POLICY_LABEL[r["policy"]] + (" **(cheaper fixed control)**" if r["policy"] == "k50" else "")
            if not r["available"]:
                L.append(f"| {lab} | — | — | missing |")
                continue
            if r["policy"] == ma.POLICY:
                dtxt = "—"
            elif r.get("paired"):
                dtxt = f"{r['diff_vs_adaptive']:+.3f} [{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}]"
            else:
                dtxt = "not paired (different seeds): rates only"
            L.append(f"| {lab} | {_rate(r)} | {r['mean_calls']:.2f} | {dtxt} |")
        L.append("")

    L += ["## Re-plans of the adaptive rule (descriptive)", "",
          "Per episode: number of diff-triggered re-plans, block-exhausted re-plans, comparisons (checks) and policy calls "
          "(`n_policy_calls = 1 + n_replans_exhaust + n_checks`).", "",
          "| condition | n | mean diff-triggered re-plans | median (max) | episodes with ≥1 | distribution 0 / 1 / 2 / 3 / 4+ | mean block-exhausted | mean checks | mean policy calls |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in ctx["replans"]:
        h = r["hist_diff_replans"]
        L.append(f"| {r['condition']} | {r['n']} | {r['mean_diff_replans']:.2f} | {r['median_diff_replans']:.0f} ({r['max_diff_replans']}) | "
                 f"{100 * r['frac_with_diff_replan']:.1f}% | {h['0']} / {h['1']} / {h['2']} / {h['3']} / {h['4+']} | "
                 f"{r['mean_exhaust_replans']:.2f} | {r['mean_checks']:.2f} | {r['mean_policy_calls']:.2f} |")
    L.append("")

    L += ["## Reaction to the push (descriptive; cell push 0.06 m only)", ""]
    rc = ctx["reaction"]
    if not rc:
        L.append("No push-cell data.")
    elif not rc["available"]:
        L.append(f"Not computed: {rc['reason']}.")
    else:
        L += [f"Episodes with an applied push (`push_status = pushed`): {rc['n_pushed']}. For each, the number of steps from the push step to the first "
              "**diff-triggered** re-plan strictly after it (the push is applied at the end of step `push_step`, after the policy chose that step's "
              "action, so the policy first sees the pushed scene at `push_step + 1`; block-exhausted re-plans are not reactions).", "",
              f"- Episodes with a diff-triggered re-plan after the push: {rc['n_with_reaction']}; without (or ended first): {rc['n_no_reaction']}.",
              f"- Episodes in which a diff-triggered re-plan happened exactly at `push_step` (observation before the push, not counted as reaction): "
              f"{rc['n_replan_at_push_step']}."]
        if rc["n_with_reaction"]:
            fw = rc["frac_within"]
            L += [f"- Steps to the first reaction: min {rc['min']}, Q1 {rc['q1']:.1f}, median {rc['median']:.1f}, Q3 {rc['q3']:.1f}, max {rc['max']}; "
                  f"within 10 / 20 / 30 steps: {100 * fw['10']:.1f}% / {100 * fw['20']:.1f}% / {100 * fw['30']:.1f}% of the episodes with a reaction."]
    L.append("")

    L += ["## Secondary metrics of the adaptive cells (descriptive only, no tests)", "",
          f"Mean over all episodes with bootstrap 95% CI ({N_BOOT} resamples over episodes, RNG seed {BOOT_SEED}).", "",
          "| cell | " + " | ".join(t for _, t, _ in SECONDARY) + " |", "|---|" + "---|" * len(SECONDARY)]
    for r in ctx["secondary"]:
        L.append(f"| {r['cell']} | " + " | ".join(
            "—" if f"{col}_mean" not in r else f"{r[f'{col}_mean']:.{nd}f} [{r[f'{col}_lo']:.{nd}f}, {r[f'{col}_hi']:.{nd}f}]"
            for col, _, nd in SECONDARY) + " |")
    L.append("")
    pa = ctx["push"]
    if pa:
        L += ["Push episodes in `ad__push0.06` (all included): " + f"n = {pa['n']}, pushed {pa['pushed']}, skipped_lifted {pa['skipped_lifted']}, "
              f"truncated {pa['truncated']} (table {pa['truncated_table']} / arm {pa['truncated_arm']}), not_reached {pa['not_reached']}.", ""]
    L += ["Figure: `fig_adaptive.png` (left: HA1/HA2 differences with 95% CI; middle: success vs policy calls; right: steps from push to first diff-triggered re-plan).", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- 图
def plot_figure(ctx: dict, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = "DRY RUN (not results)" if ctx["dry_run"] else f"seeds {ctx['seeds_label']}"
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1.1, 1.0]))
    # (1) HA1 / HA2
    ax = axes[0]
    items = [(ctx["hyp"][h]) for h in ("HA2", "HA1") if ctx["hyp"][h]["available"]]
    for i, r in enumerate(items):
        ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r["ci_lo"]], [r["ci_hi"] - r["diff"]]], fmt="o", color="#0072B2", capsize=3)
    ax.axvline(0, color="0.4", lw=0.8)
    if any(r["id"] == "HA2" for r in items):
        ax.axvline(-ma.NONINF_MARGIN, color="#D55E00", lw=1.0, ls="--")
        ax.text(-ma.NONINF_MARGIN, len(items) - 0.45, " margin −0.05", color="#D55E00", fontsize=7, va="bottom")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f"{r['id']} ({r['condition']})" for r in items], fontsize=8)
    ax.set_ylim(-0.6, max(len(items) - 0.2, 0.8))
    ax.set_xlabel("success difference adaptive − k=100\n(paired bootstrap 95% CI)", fontsize=8)
    ax.set_title("Primary hypotheses", fontsize=9)
    # (2) success vs calls
    ax = axes[1]
    colors = {"k20": "#009E73", "k50": "#E69F00", "k100": "#D55E00", "te": "#000000", "ad": "#0072B2"}
    marks = {"none": "o", "push0.06": "s"}
    for r in ctx["desc"]:
        if not r["available"]:
            continue
        ax.errorbar(r["mean_calls"], r["rate"], yerr=yerr_from_ci([r["rate"]], [r["wilson_lo"]], [r["wilson_hi"]]), fmt=marks[r["condition"]],
                    color=colors[r["policy"]], ms=6, capsize=2, elinewidth=0.8)
        ax.annotate(POLICY_LABEL[r["policy"]].replace("temporal ensemble", "te"), (r["mean_calls"], r["rate"]), fontsize=6, xytext=(4, 3),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("mean policy calls per episode (log)", fontsize=8)
    ax.set_ylabel("success rate (Wilson 95% CI)", fontsize=8)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("EXPLORATORY: success vs cost (o none, □ push 0.06)", fontsize=8)
    # (3) reaction histogram
    ax = axes[2]
    rc = ctx["reaction"]
    if rc and rc.get("available") and rc["steps"]:
        s = np.array(rc["steps"])
        bins = np.arange(0, max(s.max(), 40) + 10, 10)
        ax.hist(s, bins=bins, color="#0072B2", edgecolor="white")
        ax.set_xlabel("steps from push to first diff-triggered re-plan", fontsize=8)
        ax.set_ylabel("episodes", fontsize=8)
        ax.set_title(f"EXPLORATORY: reaction to the push\n({rc['n_with_reaction']} of {rc['n_pushed']} pushed episodes reacted)", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no reaction data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle(f"Adaptive re-observation rule, pretrained ACT, {tag}", fontsize=10)
    fig.tight_layout()
    p = out_dir / "fig_adaptive.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def _csv_rows(rows, cols, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for t in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in t.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/main_adaptive", help="含 <ad-policy>__none / <ad-policy>__push0.06 的 episodes.csv 的目录")
    ap.add_argument("--ad-policy", default=ma.POLICY, help="自适应格子的名字前缀（默认 ad；演示运行用 ad_m10_q90）")
    ap.add_argument("--out", default=None, help="输出目录（默认 = --dir）")
    ap.add_argument("--seeds", default=ma.ADAPTIVE_SEEDS)
    ap.add_argument("--control-dir", default=ma.CONTROL_DIR, help="对照 k=100 所在目录（格子名 k100__none 等，同种子）")
    ap.add_argument("--extra-control-dir", action="append", default=[], help="k20/k50/te（以及找不到的对照）额外去哪些目录找（可重复）")
    ap.add_argument("--dry-run", action="store_true", help="强制标 DRY RUN（演示运行）")
    a = ap.parse_args(argv)
    d, cd = _abs(a.dir), _abs(a.control_dir)
    out = _abs(a.out) if a.out else d
    out.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(a.seeds)
    reg = ma.registered_params(ROOT)
    search = [cd] + [_abs(x) for x in a.extra_control_dir]

    ad = {tag: load_cell(d, f"{a.ad_policy}__{tag}", seeds) for tag, _, _ in ma.CONDITIONS}
    fixed = {pol: {tag: find_fixed(pol, tag, search, seeds) for tag, _, _ in ma.CONDITIONS} for pol in ma.FIXED_POLICIES}
    k100 = fixed[ma.CONTROL_POLICY]
    if not any(c["rows"] for c in ad.values()):
        raise SystemExit(f"{d} 下没有 {a.ad_policy}__none / {a.ad_policy}__push0.06 的 episodes.csv")

    prov = []
    for tag, c in ad.items():
        if c["rows"]:
            ok, why = adaptive_provenance(c, reg)
            prov.append(dict(item=c["name"], ok=ok, why=why))
    bad = [p for p in prov if not p["ok"]]
    dry_run = a.dry_run or min(seeds) < 200000 or bool(bad) or not prov
    reasons = []
    if a.dry_run:
        reasons.append("run with --dry-run.")
    if min(seeds) < 200000:
        reasons.append("tuning seeds (< 200000) were used.")
    if bad:
        reasons.append(f"{len(bad)} of {len(prov)} adaptive cells do not record exactly the registered parameters (e.g. {bad[0]['item']}: {bad[0]['why']}).")
    unpaired = sorted({f"{pol} ({c['source']})" for pol in ma.FIXED_POLICIES for c in fixed[pol].values() if c["rows"] and not c.get("paired")})
    unpaired_note = (f"Cells taken from other seed sets are NOT paired with the adaptive cells (different seeds): {', '.join(unpaired)}; "
                     "for them only success rates and policy calls are listed, no paired differences." if unpaired else "")

    ctx = dict(dry_run=dry_run, dry_reason=" ".join(reasons), unpaired_note=unpaired_note, provenance=prov, reg=reg, n=len(seeds),
               seeds_label=a.seeds, control_label=str(a.control_dir),
               issues=[f"{c['name']}: {c['issue']}" for c in list(ad.values()) + list(k100.values()) if c["rows"] and c["issue"]])
    ctx["hyp"] = run_hypotheses(ad, k100)
    ctx["desc"] = descriptive(ad, fixed)
    ctx["replans"] = replan_stats(ad)
    ctx["secondary"] = secondary_table(ad)
    ctx["push"] = push_accounting(ad.get("push0.06"))
    ctx["reaction"] = reaction(ad.get("push0.06"))

    payload = dict(dry_run=dry_run, dry_reason=ctx["dry_reason"], registered_params=reg, **{k: ctx["hyp"][k] for k in ("HA1", "HA2", "conclusion")},
                   reaction=ctx["reaction"], replans=ctx["replans"])
    (out / "hypotheses_adaptive.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    cols = ["condition", "policy", "available", "n", "n_success", "rate", "wilson_lo", "wilson_hi", "mean_calls", "paired",
            "diff_vs_adaptive", "diff_lo", "diff_hi"]
    _csv_rows(ctx["desc"], cols, out / "main_adaptive_cells.csv")
    fig = plot_figure(ctx, out)
    (out / "main_adaptive_report.md").write_text(build_report(ctx), encoding="utf-8", newline="\n")

    print(f"[analyze_main_adaptive] {'DRY RUN ' if dry_run else ''}-> {out / 'main_adaptive_report.md'}、hypotheses_adaptive.json、"
          f"main_adaptive_cells.csv、{fig.name}")
    for hid in ("HA1", "HA2"):
        r = ctx["hyp"][hid]
        if r["available"]:
            print(f"  {hid} {r['condition']} adaptive−k100: {r['rate_A']:.3f} vs {r['rate_B']:.3f}, diff {r['diff']:+.3f} "
                  f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] p(McNemar)={r['p_mcnemar']:.4g} -> {r['decision']}")
        else:
            print(f"  {hid}: not testable ({r['reason']})")
    print(f"  结论：{ctx['hyp']['conclusion']['text']}")
    if ctx["issues"]:
        print(f"[analyze_main_adaptive] 不完整的格子 {len(ctx['issues'])} 个（见报告开头）")


if __name__ == "__main__":
    main()
