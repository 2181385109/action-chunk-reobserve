"""T9 DP 正式分析：严格按 PREREGISTRATION_DP.md 实现 H4–H5 的检验、次要指标描述、探索性图表。

用法（WSL 内；run_main_dp.sh 在 12 格全部完成后自动调用）：
  python scripts/analyze_main_dp.py [--dir results/main_dp] [--out results/main_dp] [--seeds 200000-200199]
演示运行（调参种子，不是结果）：
  python scripts/analyze_main_dp.py --dir results/pilot_dp --out results/pilot_dp/analyze_dryrun/main_dp --seeds 5000-5049

输出（--out）：main_dp_report.md、main_dp_cells.csv（12 格的成功率 / 次要指标）、hypotheses_dp.json（H4–H5）、
  fig_hypotheses_dp.png（H4–H5 的成功率差 + 95% 区间）、fig_success_dp.png（探索性：12 格成功率）。
种子里有 <200000 的就是演示运行，报告开头会明确标注“DRY RUN，不是结果”。
所有回合都计入，不剔除；某格回合数 / 种子集合与登记的不一致时，该格标为不完整，涉及它的检验不做。
检验、bootstrap、Holm、“相反结果”与 PREREGISTRATION.md 相同（chunkre.stats）；Holm 族是 H4、H5 两个。
“差距变化”（探索性）在 analyze_interaction.py。
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

from analyze_main import _fmt, _num, load_cell   # noqa: E402  （读一格 / 格式化：与 ACT 分析共用）
from chunkre import main_dp as md  # noqa: E402
from chunkre.config import parse_seeds  # noqa: E402
from chunkre.metrics import wilson_ci  # noqa: E402
from chunkre.plotting import yerr_from_ci  # noqa: E402
from chunkre.stats import (ALPHA, BOOT_SEED, N_BOOT, bootstrap_mean_ci, classify, holm_adjust,  # noqa: E402
                           mcnemar_exact, pair_counts, paired_bootstrap_diff)

# (列, 标题, 小数位)：登记的四个次要指标 + DP 另报的平均最大覆盖率
SECONDARY = [("n_policy_calls", "policy calls per episode", 1), ("jerk", "jerk", 4),
             ("path_length", "path_length", 2), ("smoothness", "smoothness", 4),
             ("max_reward", "mean maximum coverage (max_reward)", 3)]
POLICY_LABEL = {"k4": "k=4", "k8": "k=8", "k15": "k=15"}
STYLE = {"k4": ("#D55E00", "D", "-"), "k8": ("#E69F00", "^", "-"), "k15": ("#009E73", "s", "-")}   # Okabe-Ito
NAN = float("nan")


def load_all(cell_dir: Path, expected_seeds: list[int]) -> dict[str, dict]:
    return {c.name: load_cell(cell_dir, c.name, expected_seeds) for c in md.cells()}


def run_hypotheses(data: dict[str, dict]) -> list[dict]:
    """H4–H5：按种子配对的精确 McNemar（双侧）+ 成功率差与配对 bootstrap 区间；两个 p 值一起做 Holm 校正。"""
    res = []
    for hid, tag, A, B, acts in md.HYPOTHESES:
        ca, cb = data.get(f"{A}__{tag}"), data.get(f"{B}__{tag}")
        r = dict(id=hid, act_analogue=acts, condition=tag, A=A, B=B, claim=f"success rate of {A} > {B} under {tag}")
        if not (ca and cb and ca["complete"] and cb["complete"]):
            r.update(available=False, reason="; ".join(f"{c['name']}: {c['issue']}" for c in (ca, cb) if c and not c["complete"]))
            res.append(r)
            continue
        a, b = ca["success"], cb["success"]
        pc = pair_counts(a, b)
        diff, lo, hi = paired_bootstrap_diff(a, b)
        r.update(available=True, n=int(a.size), rate_A=float(a.mean()), rate_B=float(b.mean()), diff=diff, ci_lo=lo,
                 ci_hi=hi, **pc, p_raw=mcnemar_exact(pc["only_a"], pc["only_b"]))
        res.append(r)
    av = [r for r in res if r["available"]]
    if len(av) == len(res):   # Holm 只在两个假设都能检验时才有意义
        for r, adj in zip(res, holm_adjust([r["p_raw"] for r in res])):
            r["p_holm"] = adj
            r["decision"] = classify(r["diff"], adj)
    else:
        for r in av:
            r["p_holm"], r["decision"] = NAN, "not computed (not all hypotheses testable)"
    return res


def cell_table(data: dict[str, dict]) -> list[dict]:
    """每格：成功率 + Wilson 区间、次要指标均值 + bootstrap 区间（对所有回合，成功失败混合；NaN 回合不计入该指标）。"""
    out = []
    for c in md.cells():
        d = data[c.name]
        row = dict(cell=c.name, policy=c.policy, condition=c.name.split("__", 1)[1], strength=c.strength,
                   complete=d["complete"], n=len(d["rows"]))
        if d["rows"]:
            s = d["success"]
            lo, hi = wilson_ci(int(s.sum()), len(s))
            row.update(n_success=int(s.sum()), success_rate=float(s.mean()), wilson_lo=lo, wilson_hi=hi)
            for col, _, _ in SECONDARY:
                m, mlo, mhi, nv = bootstrap_mean_ci([_num(r.get(col)) for r in d["rows"]])
                row.update({f"{col}_mean": m, f"{col}_lo": mlo, f"{col}_hi": mhi, f"{col}_n": nv})
        out.append(row)
    return out


def push_accounting(data: dict[str, dict]) -> list[dict]:
    """推 T 形块各格的实际情况（都照常计入分析，只是单独报告）：推了 / 重叠跳过 / 推之前已成功 / 撞墙截断 / 重抽次数。"""
    out = []
    for c in md.cells():
        if c.kind != "push" or not data[c.name]["rows"]:
            continue
        rows = data[c.name]["rows"]
        st = [r.get("push_status", "") for r in rows]
        clip = [r.get("push_clip", "") for r in rows]
        rd = [int(float(r["push_redraws"])) for r in rows if r.get("push_redraws") not in ("", None)]
        out.append(dict(cell=c.name, n=len(rows), pushed=st.count("pushed"), skipped_overlap=st.count("skipped_overlap"),
                        not_reached=st.count("not_reached"), zero=st.count("zero"),
                        truncated_wall=sum(1 for x in clip if "wall" in x),
                        redrawn=sum(1 for x in rd if x > 0), max_redraws=(max(rd) if rd else 0),
                        mean_redraws=(float(np.mean(rd)) if rd else NAN)))
    return out


def best_policies(table: list[dict]) -> list[dict]:
    """探索性：每个条件下成功率最高的 k（并列全列出）和领先第二名多少个百分点。只描述，不检验。"""
    out = []
    for tag in md.condition_tags():
        rows = [t for t in table if t["condition"] == tag and "success_rate" in t]
        if not rows:
            continue
        rows.sort(key=lambda t: -t["success_rate"])
        top = rows[0]["success_rate"]
        best = [t["policy"] for t in rows if abs(t["success_rate"] - top) < 1e-12]
        rest = [t for t in rows if t["policy"] not in best]
        out.append(dict(condition=tag, best=best, rate=top, runner_up=(rest[0]["policy"] if rest else None),
                        gap_pts=(100 * (top - rest[0]["success_rate"]) if rest else NAN)))
    return out


def _pivot(table, key_fn):
    by = {t["cell"]: t for t in table}
    pols = [p for p, _, _ in md.POLICIES]
    L = ["| condition | " + " | ".join(POLICY_LABEL[p] for p in pols) + " |", "|---|" + "---|" * len(pols)]
    for tag in md.condition_tags():
        L.append(f"| {tag} | " + " | ".join(key_fn(by.get(f"{p}__{tag}")) for p in pols) + " |")
    return L


def build_report(hyp, table, push, best, expected_n, seeds_label, dry_run, issues) -> str:
    L = []
    if dry_run:
        L += ["> **DRY RUN — NOT RESULTS.** This report was produced from the tuning-seed pilot data "
              f"(seeds {seeds_label}, n = {expected_n} per cell) only to check that the analysis code runs. "
              "Nothing here is a finding.", ""]
    L += ["# DP main analysis report (T9, implements PREREGISTRATION_DP.md)", ""]
    L.append(f"- Cells: {sum(1 for t in table if t['complete'])}/{len(table)} complete "
             f"(expected {expected_n} episodes per cell, seeds {seeds_label}, identical seed set in every cell).")
    L.append(f"- Tests: exact two-sided McNemar on seed-paired outcomes; paired bootstrap 95% CI "
             f"({N_BOOT} resamples, RNG seed {BOOT_SEED}); Holm correction over H4–H5 only (alpha = {ALPHA}); "
             "separate from the ACT family H1–H3.")
    for i in issues:
        L.append(f"- **INCOMPLETE** {i}")
    L.append("")

    L += ["## Primary hypotheses (confirmatory)", ""]
    L += ["| H | (ACT analogue) | condition | A vs B | n | rate A | rate B | diff (A−B) | 95% CI (paired bootstrap) | only A / only B | p (McNemar) | p (Holm) | decision |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    names = {"supported": "supported", "opposite": "OPPOSITE RESULT", "not_significant": "not significant"}
    for r in hyp:
        if not r["available"]:
            L.append(f"| {r['id']} | {r['act_analogue']} | {r['condition']} | {r['A']} vs {r['B']} | — | — | — | — | — | — | — | — | not testable: {r['reason']} |")
            continue
        L.append(f"| {r['id']} | {r['act_analogue']} | {r['condition']} | {POLICY_LABEL[r['A']]} vs {POLICY_LABEL[r['B']]} | {r['n']} | "
                 f"{r['rate_A']:.3f} | {r['rate_B']:.3f} | {r['diff']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | "
                 f"{r['only_a']} / {r['only_b']} | {r['p_raw']:.4g} | {_fmt(r.get('p_holm', NAN), 4)} | "
                 f"{names.get(r.get('decision'), r.get('decision', '—'))} |")
    L += ["", "Hypothesis H: success rate of A is higher than that of B (two-sided test). "
              "\"only A / only B\" = seeds where only A / only B succeeded (the discordant pairs McNemar uses). "
              "\"OPPOSITE RESULT\" = difference in the opposite direction and Holm-adjusted p < 0.05; reported as such. "
              "The PREREGISTRATION_DP.md §7 states in advance that an opposite result for H4 is realistic "
              "(k=4 is far below k=15 already without perturbation); the exploratory gap change "
              "(`analyze_interaction.py`) separates the baseline gap from the effect of the perturbation.", ""]

    L += ["## Success rate, all 12 cells (exploratory description; Wilson 95% CI)", ""]
    L += _pivot(table, lambda t: "—" if not t or "success_rate" not in t else
                f"{t['n_success']}/{t['n']} = {t['success_rate']:.3f} [{t['wilson_lo']:.3f}, {t['wilson_hi']:.3f}]"
                + ("" if t["complete"] else " (incomplete)"))
    L.append("")

    L += ["## Secondary metrics (descriptive only, no tests)", "",
          "Mean over all episodes of the cell (successes and failures together; episodes where a metric is undefined "
          "are left out of that metric's mean) with bootstrap 95% CI "
          f"({N_BOOT} resamples over episodes, RNG seed {BOOT_SEED}). "
          "path_length and jerk are per-episode quantities of the executed action sequence; smoothness is the mean "
          "L2 norm of consecutive action differences; max_reward is the episode's maximum coverage score (0–1).", ""]
    for col, title, nd in SECONDARY:
        L += [f"### {title}", ""]
        L += _pivot(table, lambda t, col=col, nd=nd: "—" if not t or f"{col}_mean" not in t else
                    f"{t[f'{col}_mean']:.{nd}f} [{t[f'{col}_lo']:.{nd}f}, {t[f'{col}_hi']:.{nd}f}]")
        L.append("")

    L += ["## Push episodes: pushed / skipped / truncated / redraw counts (all included in the analysis)", "",
          "pushed = the push was applied; skipped_overlap = all 20 redrawn directions overlapped the agent, no push applied; "
          "not_reached = the episode had already succeeded before the push step; truncated_wall = the displacement was "
          "shortened so the block stays inside the walls; redrawn = episodes in which the direction was redrawn at least once.", "",
          "| cell | n | pushed | skipped_overlap | not_reached | truncated_wall | redrawn (max / mean redraws) |",
          "|---|---|---|---|---|---|---|"]
    for p in push:
        L.append(f"| {p['cell']} | {p['n']} | {p['pushed']} | {p['skipped_overlap']} | {p['not_reached']} | "
                 f"{p['truncated_wall']} | {p['redrawn']} ({p['max_redraws']} / {_fmt(p['mean_redraws'], 2)}) |")
    L.append("")

    L += ["## Best k per condition (EXPLORATORY; description only, no test)", "",
          "| condition | best k | success rate | runner-up | gap (pct. points) |", "|---|---|---|---|---|"]
    for b in best:
        L.append(f"| {b['condition']} | {', '.join(POLICY_LABEL[x] for x in b['best'])} | {b['rate']:.3f} | "
                 f"{POLICY_LABEL[b['runner_up']] if b['runner_up'] else '—'} | {_fmt(b['gap_pts'], 1)} |")
    L += ["", "Figures: `fig_hypotheses_dp.png` (confirmatory), `fig_success_dp.png` (exploratory success rates).", ""]
    return "\n".join(L)


def plot_figures(hyp, table, out_dir: Path, seeds_label: str, dry_run: bool) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = "DRY RUN (tuning seeds, not results)" if dry_run else f"seeds {seeds_label}"
    paths = []

    # 1) H4–H5：成功率差 + 配对 bootstrap 区间
    av = [r for r in hyp if r["available"]]
    fig, ax = plt.subplots(figsize=(6.4, 2.0 + 0.7 * max(1, len(av))))
    for i, r in enumerate(av):
        y = len(av) - 1 - i
        ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r["ci_lo"]], [r["ci_hi"] - r["diff"]]], fmt="o", color="#0072B2",
                    capsize=3)
        ax.text(1.02, y, f"p_Holm = {_fmt(r.get('p_holm', NAN), 3)}", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(range(len(av)))
    ax.set_yticklabels([f"{r['id']}: {r['A']} − {r['B']}\n({r['condition']})" for r in reversed(av)], fontsize=8)
    ax.set_xlabel("difference in success rate (A − B), paired bootstrap 95% CI")
    ax.set_title(f"Primary hypotheses, PushT Diffusion Policy, {tag}", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    p = out_dir / "fig_hypotheses_dp.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # 2) 探索性：12 格成功率（横轴 = 条件，每个 k 一条线）
    by = {t["cell"]: t for t in table}
    tags = md.condition_tags()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    nudge = np.linspace(-1, 1, len(md.POLICIES)) * 0.08
    for i, (pol, _, _) in enumerate(md.POLICIES):
        col, mk, ls = STYLE[pol]
        pts = [(j + nudge[i], by[f"{pol}__{t}"]["success_rate"], by[f"{pol}__{t}"]["wilson_lo"], by[f"{pol}__{t}"]["wilson_hi"])
               for j, t in enumerate(tags) if f"{pol}__{t}" in by and "success_rate" in by[f"{pol}__{t}"]]
        if not pts:
            continue
        px, py, lo, hi = (np.array([q[j] for q in pts]) for j in range(4))
        ax.errorbar(px, py, yerr=yerr_from_ci(py, lo, hi), color=col, marker=mk, ls=ls, lw=1.6, ms=5, capsize=2,
                    elinewidth=0.9, label=POLICY_LABEL[pol])
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags)
    ax.set_xlabel("condition")
    ax.set_ylabel("success rate (Wilson 95% CI)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"EXPLORATORY: PushT Diffusion Policy, {tag}", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    p = out_dir / "fig_success_dp.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    return paths


def write_cells_csv(table, path: Path):
    cols = ["cell", "policy", "condition", "strength", "complete", "n", "n_success", "success_rate", "wilson_lo", "wilson_hi"]
    for col, _, _ in SECONDARY:
        cols += [f"{col}_mean", f"{col}_lo", f"{col}_hi", f"{col}_n"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for t in table:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in t.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/main_dp", help="含 <格子名>/episodes.csv 的目录")
    ap.add_argument("--out", default=None, help="输出目录（默认 = --dir）")
    ap.add_argument("--seeds", default=md.MAIN_DP_SEEDS, help="预期种子（默认正式种子 200000-200199）")
    a = ap.parse_args(argv)
    d = Path(a.dir) if Path(a.dir).is_absolute() else ROOT / a.dir
    out = Path(a.out) if a.out else d
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(a.seeds)
    dry_run = min(seeds) < 200000

    data = load_all(d, seeds)
    if not any(v["rows"] for v in data.values()):
        raise SystemExit(f"{d} 下没有任何格子的 episodes.csv")
    issues = [f"{k}: {v['issue']}" for k, v in data.items() if not v["complete"]]
    hyp = run_hypotheses(data)
    table = cell_table(data)
    push = push_accounting(data)
    best = best_policies(table)

    (out / "hypotheses_dp.json").write_text(json.dumps(hyp, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    write_cells_csv(table, out / "main_dp_cells.csv")
    figs = plot_figures(hyp, table, out, a.seeds, dry_run)
    report = build_report(hyp, table, push, best, len(seeds), a.seeds, dry_run, issues)
    (out / "main_dp_report.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"[analyze_main_dp] {'DRY RUN ' if dry_run else ''}{sum(v['complete'] for v in data.values())}/{len(data)} 格完整 -> "
          f"{out / 'main_dp_report.md'}、main_dp_cells.csv、hypotheses_dp.json、{', '.join(f.name for f in figs)}")
    for r in hyp:
        if r["available"]:
            print(f"  {r['id']} {r['condition']} {r['A']}−{r['B']}: diff {r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
                  f"p={r['p_raw']:.4g} p_Holm={_fmt(r.get('p_holm', NAN), 4)} {r.get('decision', '')}")
        else:
            print(f"  {r['id']}: not testable ({r['reason']})")
    if issues:
        print(f"[analyze_main_dp] 不完整的格子 {len(issues)} 个（见报告开头）")


if __name__ == "__main__":
    main()
