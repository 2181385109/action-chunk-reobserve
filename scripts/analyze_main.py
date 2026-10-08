"""T6b 正式分析：严格按 PREREGISTRATION.md 实现 H1–H3 的检验、次要指标描述、探索性图表。

用法（WSL 内；run_main.sh 在 40 格全部完成后自动调用）：
  python scripts/analyze_main.py [--dir results/main] [--out results/main] [--seeds 200000-200199]
演示运行（调参种子，不是结果）：
  python scripts/analyze_main.py --dir results/pilot --out results/pilot/analyze_main_dryrun --seeds 5000-5019

输出（--out）：main_report.md、main_cells.csv（40 格的成功率 / 次要指标）、hypotheses.json（H1–H3）、
  fig_hypotheses.png（H1–H3 的成功率差 + 95% 区间）、fig_push.png / fig_noise.png / fig_delay.png（探索性：成功率曲线）。
种子里有 <200000 的就是演示运行，报告开头会明确标注“DRY RUN，不是结果”。
所有回合都计入，不剔除；某格回合数 / 种子集合与登记的不一致时，该格标为不完整，涉及它的检验不做。
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

from chunkre import main_grid as mg  # noqa: E402
from chunkre.config import parse_seeds  # noqa: E402
from chunkre.metrics import read_episodes, wilson_ci  # noqa: E402
from chunkre.plotting import yerr_from_ci  # noqa: E402
from chunkre.stats import (ALPHA, BOOT_SEED, N_BOOT, bootstrap_mean_ci, classify, holm_adjust,  # noqa: E402
                           mcnemar_exact, pair_counts, paired_bootstrap_diff)

SECONDARY = [("n_policy_calls", "policy calls per episode", 1), ("jerk", "jerk", 4),
             ("path_length", "path_length", 2), ("smoothness", "smoothness", 4)]
POLICY_LABEL = {"k20": "k=20", "k50": "k=50", "k100": "k=100", "te": "temporal ensemble"}
STYLE = {"k20": ("#009E73", "s", "-"), "k50": ("#E69F00", "^", "-"), "k100": ("#D55E00", "D", "-"),
         "te": ("#000000", "v", "--")}   # Okabe-Ito；时间集成用黑色虚线（与试跑图一致）
NAN = float("nan")


def _num(x) -> float:
    return float(x) if x not in ("", None) else NAN


def load_cell(cell_dir: Path, name: str, expected_seeds: list[int]) -> dict:
    """读一格。返回 dict：rows（逐回合）、seeds、success（0/1 数组，按种子升序）、complete（回合数和种子集合是否与登记一致）、issue。"""
    p = Path(cell_dir) / name / "episodes.csv"
    if not p.exists():
        return dict(name=name, rows=[], complete=False, issue="missing (no episodes.csv)")
    rows = sorted(read_episodes(p), key=lambda r: int(float(r["seed"])))
    seeds = [int(float(r["seed"])) for r in rows]
    complete = seeds == sorted(expected_seeds)
    issue = ""
    if not complete:
        issue = f"{len(rows)} episodes, expected {len(expected_seeds)} with seeds {expected_seeds[0]}–{expected_seeds[-1]}"
    return dict(name=name, rows=rows, seeds=seeds, success=np.array([int(float(r["success"])) for r in rows]),
                complete=complete, issue=issue)


def load_all(cell_dir: Path, expected_seeds: list[int]) -> dict[str, dict]:
    return {c.name: load_cell(cell_dir, c.name, expected_seeds) for c in mg.cells()}


def run_hypotheses(data: dict[str, dict]) -> list[dict]:
    """H1–H3：按种子配对的精确 McNemar（双侧）+ 成功率差与配对 bootstrap 区间；三个 p 值一起做 Holm 校正。"""
    res = []
    for hid, tag, A, B in mg.HYPOTHESES:
        ca, cb = data.get(f"{A}__{tag}"), data.get(f"{B}__{tag}")
        r = dict(id=hid, condition=tag, A=A, B=B, claim=f"success rate of {A} > {B} under {tag}")
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
    if len(av) == len(res):   # Holm 只在三个假设都能检验时才有意义
        for r, adj in zip(res, holm_adjust([r["p_raw"] for r in res])):
            r["p_holm"] = adj
            r["decision"] = classify(r["diff"], adj)
    else:
        for r in av:
            r["p_holm"], r["decision"] = NAN, "not computed (not all three hypotheses testable)"
    return res


def cell_table(data: dict[str, dict]) -> list[dict]:
    """每格：成功率 + Wilson 区间、次要指标均值 + bootstrap 区间（对所有回合，成功失败混合；NaN 回合不计入该指标）。"""
    out = []
    for c in mg.cells():
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
    """推方块各格的“跳过 / 截断 / 未到推的时刻”回合数（都照常计入分析，只是单独报告）。"""
    out = []
    for c in mg.cells():
        if c.kind != "push" or not data[c.name]["rows"]:
            continue
        rows = data[c.name]["rows"]
        st = [r.get("push_status", "") for r in rows]
        clip = [r.get("push_clip", "") for r in rows]
        out.append(dict(cell=c.name, n=len(rows), pushed=st.count("pushed"), skipped_lifted=st.count("skipped_lifted"),
                        not_reached=st.count("not_reached"), zero=st.count("zero"),
                        truncated=sum(1 for x in clip if x), truncated_table=sum(1 for x in clip if "table" in x),
                        truncated_arm=sum(1 for x in clip if "arm" in x)))
    return out


def best_policies(table: list[dict]) -> list[dict]:
    """探索性：每个条件下成功率最高的策略（并列全列出）和领先第二名多少个百分点。只描述，不检验。"""
    out = []
    for tag in mg.condition_tags():
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


def _fmt(x, nd=3):
    return "—" if x != x else f"{x:.{nd}f}"


def _pivot(table, key_fn):
    by = {t["cell"]: t for t in table}
    pols = [p for p, _, _ in mg.POLICIES]
    L = ["| condition | " + " | ".join(POLICY_LABEL[p] for p in pols) + " |", "|---|" + "---|" * len(pols)]
    for tag in mg.condition_tags():
        L.append(f"| {tag} | " + " | ".join(key_fn(by.get(f"{p}__{tag}")) for p in pols) + " |")
    return L


def build_report(hyp, table, push, best, expected_n, seeds_label, dry_run, issues) -> str:
    L = []
    if dry_run:
        L += ["> **DRY RUN — NOT RESULTS.** This report was produced from the tuning-seed pilot data "
              f"(seeds {seeds_label}, n = {expected_n} per cell) only to check that the analysis code runs. "
              "Nothing here is a finding.", ""]
    L += ["# Main analysis report (T6b, implements PREREGISTRATION.md)", ""]
    L.append(f"- Cells: {sum(1 for t in table if t['complete'])}/{len(table)} complete "
             f"(expected {expected_n} episodes per cell, seeds {seeds_label}, identical seed set in every cell).")
    L.append(f"- Tests: exact two-sided McNemar on seed-paired outcomes; paired bootstrap 95% CI "
             f"({N_BOOT} resamples, RNG seed {BOOT_SEED}); Holm correction over H1–H3 (alpha = {ALPHA}).")
    for i in issues:
        L.append(f"- **INCOMPLETE** {i}")
    L.append("")

    L += ["## Primary hypotheses (confirmatory)", ""]
    L += ["| H | condition | A vs B | n | rate A | rate B | diff (A−B) | 95% CI (paired bootstrap) | only A / only B | p (McNemar) | p (Holm) | decision |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    names = {"supported": "supported", "opposite": "OPPOSITE RESULT", "not_significant": "not significant"}
    for r in hyp:
        if not r["available"]:
            L.append(f"| {r['id']} | {r['condition']} | {r['A']} vs {r['B']} | — | — | — | — | — | — | — | — | not testable: {r['reason']} |")
            continue
        L.append(f"| {r['id']} | {r['condition']} | {POLICY_LABEL[r['A']]} vs {POLICY_LABEL[r['B']]} | {r['n']} | "
                 f"{r['rate_A']:.3f} | {r['rate_B']:.3f} | {r['diff']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | "
                 f"{r['only_a']} / {r['only_b']} | {r['p_raw']:.4g} | {_fmt(r.get('p_holm', NAN), 4)} | "
                 f"{names.get(r.get('decision'), r.get('decision', '—'))} |")
    L += ["", "Hypothesis H: success rate of A is higher than that of B (two-sided test). "
              "\"only A / only B\" = seeds where only A / only B succeeded (the discordant pairs McNemar uses). "
              "\"OPPOSITE RESULT\" = difference in the opposite direction and Holm-adjusted p < 0.05; reported as such.", ""]

    L += ["## Success rate, all 40 cells (exploratory description; Wilson 95% CI)", ""]
    L += _pivot(table, lambda t: "—" if not t or "success_rate" not in t else
                f"{t['n_success']}/{t['n']} = {t['success_rate']:.3f} [{t['wilson_lo']:.3f}, {t['wilson_hi']:.3f}]"
                + ("" if t["complete"] else " (incomplete)"))
    L.append("")

    L += ["## Secondary metrics (descriptive only, no tests)", "",
          "Mean over all episodes of the cell (successes and failures together; episodes where a metric is undefined "
          "are left out of that metric's mean) with bootstrap 95% CI "
          f"({N_BOOT} resamples over episodes, RNG seed {BOOT_SEED}). "
          "path_length and jerk are per-episode quantities of the executed action sequence; smoothness is the mean "
          "L2 norm of consecutive action differences.", ""]
    for col, title, nd in SECONDARY:
        L += [f"### {title}", ""]
        L += _pivot(table, lambda t, col=col, nd=nd: "—" if not t or f"{col}_mean" not in t else
                    f"{t[f'{col}_mean']:.{nd}f} [{t[f'{col}_lo']:.{nd}f}, {t[f'{col}_hi']:.{nd}f}]")
        L.append("")

    L += ["## Push episodes: skipped / truncated counts (all included in the analysis)", "",
          "skipped_lifted = the cube was already lifted at the push step, no push applied; "
          "truncated = the push distance was shortened (table edge and/or arm contact); "
          "not_reached = the episode ended before the push step.", "",
          "| cell | n | pushed | skipped_lifted | truncated (table / arm) | not_reached |", "|---|---|---|---|---|---|"]
    for p in push:
        L.append(f"| {p['cell']} | {p['n']} | {p['pushed']} | {p['skipped_lifted']} | "
                 f"{p['truncated']} ({p['truncated_table']} / {p['truncated_arm']}) | {p['not_reached']} |")
    L.append("")

    L += ["## Best policy per condition (EXPLORATORY; description only, no test)", "",
          "| condition | best policy | success rate | runner-up | gap (pct. points) |", "|---|---|---|---|---|"]
    for b in best:
        L.append(f"| {b['condition']} | {', '.join(POLICY_LABEL[x] for x in b['best'])} | {b['rate']:.3f} | "
                 f"{POLICY_LABEL[b['runner_up']] if b['runner_up'] else '—'} | {_fmt(b['gap_pts'], 1)} |")
    L += ["", "Figures: `fig_hypotheses.png` (confirmatory), `fig_push.png`, `fig_noise.png`, `fig_delay.png` (exploratory success-rate curves).", ""]
    return "\n".join(L)


def plot_figures(hyp, table, out_dir: Path, seeds_label: str, dry_run: bool) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = "DRY RUN (tuning seeds, not results)" if dry_run else f"seeds {seeds_label}"
    paths = []

    # 1) H1–H3：成功率差 + 配对 bootstrap 区间
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
    ax.set_title(f"Primary hypotheses, aloha ACT, {tag}", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    p = out_dir / "fig_hypotheses.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # 2) 探索性：每种扰动一张成功率曲线
    by = {t["cell"]: t for t in table}
    xlabel = {"push": "push distance (m)", "noise": "image noise sigma (0-255)", "delay": "action delay (steps)"}
    for short, kind, levels in mg.PERTURBS:
        if kind == "none":
            continue
        xs = [0.0] + [float(s) for s in levels]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        nudge = np.linspace(-1, 1, len(mg.POLICIES)) * 0.012 * (xs[-1] - xs[0])
        for i, (pol, _, _) in enumerate(mg.POLICIES):
            col, mk, ls = STYLE[pol]
            pts = []
            for x, t in zip(xs, [by.get(f"{pol}__none")] + [by.get(f"{pol}__{short}{lv:g}") for lv in levels]):
                if t and "success_rate" in t:
                    pts.append((x + nudge[i], t["success_rate"], t["wilson_lo"], t["wilson_hi"]))
            if not pts:
                continue
            px, py, lo, hi = (np.array([q[j] for q in pts]) for j in range(4))
            ax.errorbar(px, py, yerr=yerr_from_ci(py, lo, hi), color=col, marker=mk, ls=ls, lw=1.6, ms=5, capsize=2,
                        elinewidth=0.9, label=POLICY_LABEL[pol])
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs])
        ax.set_xlabel(xlabel[short] + "  (0 = no perturbation)")
        ax.set_ylabel("success rate (Wilson 95% CI)")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"EXPLORATORY: aloha ACT, {tag}: {short}", fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        p = out_dir / f"fig_{short}.png"
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
    ap.add_argument("--dir", default="results/main", help="含 <格子名>/episodes.csv 的目录")
    ap.add_argument("--out", default=None, help="输出目录（默认 = --dir）")
    ap.add_argument("--seeds", default=mg.MAIN_SEEDS, help="预期种子（默认正式种子 200000-200199）")
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

    (out / "hypotheses.json").write_text(json.dumps(hyp, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    write_cells_csv(table, out / "main_cells.csv")
    figs = plot_figures(hyp, table, out, a.seeds, dry_run)
    report = build_report(hyp, table, push, best, len(seeds), a.seeds, dry_run, issues)
    (out / "main_report.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"[analyze_main] {'DRY RUN ' if dry_run else ''}{sum(v['complete'] for v in data.values())}/{len(data)} 格完整 -> "
          f"{out / 'main_report.md'}、main_cells.csv、hypotheses.json、{', '.join(f.name for f in figs)}")
    for r in hyp:
        if r["available"]:
            print(f"  {r['id']} {r['condition']} {r['A']}−{r['B']}: diff {r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
                  f"p={r['p_raw']:.4g} p_Holm={_fmt(r.get('p_holm', NAN), 4)} {r.get('decision', '')}")
        else:
            print(f"  {r['id']}: not testable ({r['reason']})")
    if issues:
        print(f"[analyze_main] 不完整的格子 {len(issues)} 个（见报告开头）")


if __name__ == "__main__":
    main()
