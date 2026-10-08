"""T9 探索性分析：两个模型的“差距变化”（PREREGISTRATION_DP.md §6.2）。**不做显著性检验，不是证据。**

差距变化 = 每个种子的 [短 k 的成败 − 长 k 的成败]（扰动条件下）减去 [短 k − 长 k]（无扰动下），取平均，
给配对 bootstrap 95% 区间（按种子重抽样，10000 次，随机种子 20261007）。正值 = 短 k 在扰动下相对少吃亏。
- ACT：k=20（短）对 k=100（长），条件 push0.06 / noise16 / delay10，读 results/main/。**这个分析是看过 ACT 数据之后才加的（事后）。**
- DP ：k=4 （短）对 k=15 （长），条件 push60 / delay4，读 results/main_dp/。

用法（run_main_dp.sh 跑完 12 格后自动调用）：
  python scripts/analyze_interaction.py [--act-dir results/main] [--dp-dir results/main_dp] [--out results/main_dp/interaction]
          [--act-seeds 200000-200199] [--dp-seeds 200000-200199]
演示运行（DP 用试跑数据，ACT 用已有的正式数据）：
  python scripts/analyze_interaction.py --dp-dir results/pilot_dp --dp-seeds 5000-5049 --out results/pilot_dp/analyze_dryrun/interaction
某个模型的种子里有 <200000 的，该模型的部分标 DRY RUN。某格回合数 / 种子集合不对（缺格）时，涉及它的行不算，报告里写明原因。

输出（--out）：interaction_report.md、interaction.csv、interaction.json、fig_interaction.png。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_main import load_cell   # noqa: E402
from chunkre.config import parse_seeds  # noqa: E402
from chunkre.stats import ALPHA, BOOT_SEED, N_BOOT, gap_change  # noqa: E402

NAN = float("nan")
# 登记的比较（PREREGISTRATION_DP.md §6.2）：模型 -> (短 k 的格子前缀, 长 k 的格子前缀, 扰动条件标签)
MODELS = {
    "ACT": dict(short="k20", long="k100", conds=["push0.06", "noise16", "delay10"], title="ACT (aloha transfer-cube)",
                post_hoc=True),
    "DP": dict(short="k4", long="k15", conds=["push60", "delay4"], title="Diffusion Policy (PushT)", post_hoc=False),
}


def compute_model(model: str, cell_dir: Path, seeds: list[int]) -> list[dict]:
    spec = MODELS[model]
    S, L = spec["short"], spec["long"]
    cache: dict[str, dict] = {}

    def get(prefix: str, tag: str) -> dict:
        name = f"{prefix}__{tag}"
        if name not in cache:
            cache[name] = load_cell(cell_dir, name, seeds)
        return cache[name]

    rows = []
    for tag in spec["conds"]:
        r = dict(model=model, short=S, long=L, condition=tag)
        needed = [get(S, tag), get(L, tag), get(S, "none"), get(L, "none")]
        bad = [c for c in needed if not c["complete"]]
        if bad:
            r.update(available=False, reason="; ".join(f"{c['name']}: {c['issue']}" for c in bad))
        else:
            g = gap_change(needed[0]["success"], needed[1]["success"], needed[2]["success"], needed[3]["success"])
            r.update(available=True, **g)
        rows.append(r)
    return rows


def build_report(results: dict[str, list[dict]], labels: dict[str, dict]) -> str:
    L = ["# Exploratory analysis: gap change under perturbation (T9; PREREGISTRATION_DP.md §6.2)", "",
         "**EXPLORATORY — no significance test, not evidence.** For each seed, "
         "gap change = [success(short k, perturbed) − success(long k, perturbed)] − [success(short k, no perturbation) − "
         "success(long k, no perturbation)], averaged over the seeds, with a paired bootstrap 95% CI "
         f"(resampling seeds, {N_BOOT} resamples, RNG seed {BOOT_SEED}). "
         "Positive = the short k loses less (or gains more) from the perturbation than the long k does; "
         "negative = the opposite. It is the change in the short-minus-long success gap caused by the perturbation, "
         "so a short k that is simply worse at baseline does not by itself produce a nonzero value.", ""]
    for model, rows in results.items():
        info = labels[model]
        spec = MODELS[model]
        L += [f"## {spec['title']}: k={spec['short'][1:]} (short) vs k={spec['long'][1:]} (long)", ""]
        if info["dry_run"]:
            L += [f"> **DRY RUN — NOT RESULTS.** Computed from tuning-seed pilot data (seeds {info['seeds']}); only checks that the code runs.", ""]
        if spec["post_hoc"]:
            L += ["> This ACT analysis was added **after the ACT data had been seen** (post hoc); it carries no confirmatory weight.", ""]
        L += [f"Data: `{info['dir']}`, seeds {info['seeds']}.", "",
              "| condition | n | short, perturbed | long, perturbed | gap (perturbed) | short, none | long, none | gap (none) | **gap change** | 95% CI (paired bootstrap) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if not r["available"]:
                L.append(f"| {r['condition']} | — | — | — | — | — | — | — | not computed: {r['reason']} | — |")
                continue
            L.append(f"| {r['condition']} | {r['n']} | {r['rate_short_pert']:.3f} | {r['rate_long_pert']:.3f} | "
                     f"{r['gap_pert']:+.3f} | {r['rate_short_none']:.3f} | {r['rate_long_none']:.3f} | {r['gap_none']:+.3f} | "
                     f"**{r['gap_change']:+.3f}** | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] |")
        L.append("")
    L += ["Figure: `fig_interaction.png`. The intervals are descriptive; they are not adjusted for multiplicity and "
          f"no decision (alpha = {ALPHA}) is drawn from them.", ""]
    return "\n".join(L)


def plot(results: dict[str, list[dict]], labels: dict[str, dict], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [(m, r) for m, rows in results.items() for r in rows if r["available"]]
    fig, ax = plt.subplots(figsize=(7.0, 1.8 + 0.5 * max(1, len(pts))))
    color = {"ACT": "#0072B2", "DP": "#D55E00"}
    for i, (m, r) in enumerate(pts):
        y = len(pts) - 1 - i
        ax.errorbar(r["gap_change"], y, xerr=[[r["gap_change"] - r["ci_lo"]], [r["ci_hi"] - r["gap_change"]]], fmt="o",
                    color=color[m], capsize=3)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(range(len(pts)))
    ax.set_yticklabels([f"{m}: k{MODELS[m]['short'][1:]} vs k{MODELS[m]['long'][1:]}, {r['condition']}"
                        + (" [dry run]" if labels[m]["dry_run"] else "") for m, r in reversed(pts)], fontsize=8)
    ax.set_xlabel("gap change = (short−long, perturbed) − (short−long, none)", fontsize=9)
    ax.set_title("EXPLORATORY: change in the short-vs-long k success gap\n(paired bootstrap 95% CI, no test)", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    p = out / "fig_interaction.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def write_csv(results, path: Path):
    cols = ["model", "short", "long", "condition", "available", "n", "rate_short_pert", "rate_long_pert", "gap_pert",
            "rate_short_none", "rate_long_none", "gap_none", "gap_change", "ci_lo", "ci_hi", "reason"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rows in results.values():
            for r in rows:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def _abs(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else ROOT / p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--act-dir", default="results/main")
    ap.add_argument("--dp-dir", default="results/main_dp")
    ap.add_argument("--act-seeds", default="200000-200199")
    ap.add_argument("--dp-seeds", default="200000-200199")
    ap.add_argument("--out", default="results/main_dp/interaction")
    a = ap.parse_args(argv)
    out = _abs(a.out)
    out.mkdir(parents=True, exist_ok=True)

    results, labels = {}, {}
    for model, dir_, seeds_s in (("ACT", a.act_dir, a.act_seeds), ("DP", a.dp_dir, a.dp_seeds)):
        seeds = parse_seeds(seeds_s)
        labels[model] = dict(dir=dir_, seeds=seeds_s, dry_run=min(seeds) < 200000)
        results[model] = compute_model(model, _abs(dir_), seeds)

    (out / "interaction.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    write_csv(results, out / "interaction.csv")
    fig = plot(results, labels, out)
    (out / "interaction_report.md").write_text(build_report(results, labels), encoding="utf-8", newline="\n")
    print(f"[analyze_interaction] -> {out / 'interaction_report.md'}、interaction.csv、interaction.json、{fig.name}")
    for model, rows in results.items():
        tag = " (DRY RUN)" if labels[model]["dry_run"] else ""
        for r in rows:
            if r["available"]:
                print(f"  {model}{tag} {r['short']} vs {r['long']} {r['condition']}: gap change {r['gap_change']:+.3f} "
                      f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] (gap perturbed {r['gap_pert']:+.3f}, none {r['gap_none']:+.3f})")
            else:
                print(f"  {model}{tag} {r['short']} vs {r['long']} {r['condition']}: not computed ({r['reason']})")


if __name__ == "__main__":
    main()
