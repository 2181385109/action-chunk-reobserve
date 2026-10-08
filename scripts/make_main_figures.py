"""Main-experiment figures for the README, read from results/main/main_cells.csv (written by analyze_main.py from the
per-episode CSVs). No number is typed in here.

  python scripts/make_main_figures.py [--dir results/main] [--out results/main] [--copy-to docs/figures]

Writes fig_main.png (success rate vs perturbation strength, 3 panels, Wilson 95% CI) and
fig_cost.png (success rate vs mean policy calls per episode, log x; colour = policy, marker = perturbation type).
With --copy-to, also copies fig_main.png, fig_cost.png, fig_hypotheses.png there.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunkre.plotting import POLICY_LABEL, POLICY_ORDER, POLICY_STYLE, yerr_from_ci  # noqa: E402

PANELS = [("push", "Cube pushed (displacement, m)"), ("noise", "Image noise (sigma, pixel units 0-255)"),
          ("delay", "Action delay (steps at 50 Hz)")]
COND_MARKER = {"none": "o", "push": "s", "noise": "^", "delay": "D"}
COND_LABEL = {"none": "no perturbation", "push": "cube push", "noise": "image noise", "delay": "action delay"}


def read_cells(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        if r["complete"] != "True":
            raise SystemExit(f"cell {r['cell']} is incomplete; refusing to draw final figures")
        out[r["cell"]] = {k: (float(v) if k not in ("cell", "policy", "condition", "complete") and v != "" else v)
                          for k, v in r.items()}
    return out


def kind_of(cond: str) -> str:
    return "none" if cond == "none" else ("push" if cond.startswith("push") else "noise" if cond.startswith("noise") else "delay")


def fig_main(cells, out: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, (kind, xlabel) in zip(axes, PANELS):
        conds = sorted({c["condition"] for c in cells.values() if kind_of(c["condition"]) == kind},
                       key=lambda s: float(s[len(kind):]))
        xs = [0.0] + [float(c[len(kind):]) for c in conds]
        names = ["none"] + conds
        span = xs[-1] - xs[0]
        nudge = np.linspace(-1, 1, len(POLICY_ORDER)) * 0.012 * span
        for i, pol in enumerate(POLICY_ORDER):
            col, mk, ls = POLICY_STYLE[pol]
            t = [cells[f"{pol}__{n}"] for n in names]
            y = np.array([c["success_rate"] for c in t])
            lo = np.array([c["wilson_lo"] for c in t])
            hi = np.array([c["wilson_hi"] for c in t])
            ax.errorbar(np.array(xs) + nudge[i], y, yerr=yerr_from_ci(y, lo, hi), color=col, marker=mk, ls=ls, lw=1.6,
                        ms=5.5, capsize=2, elinewidth=0.9, label=POLICY_LABEL[pol])
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs])
        ax.set_xlabel(xlabel + "\n(0 = no perturbation)")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", color="0.9", lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("success rate (Wilson 95% CI)")
    axes[0].set_title("Cube pushed", fontsize=10)
    axes[1].set_title("Image noise", fontsize=10)
    axes[2].set_title("Action delay", fontsize=10)
    axes[2].legend(fontsize=8.5, frameon=False, loc="upper right")
    fig.suptitle("ACT, ALOHA transfer-cube: success rate vs perturbation strength (200 episodes per point, seeds 200000-200199)",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_cost(cells, out: Path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for c in cells.values():
        col = POLICY_STYLE[c["policy"]][0]
        ax.scatter(c["n_policy_calls_mean"], c["success_rate"], s=46, color=col, marker=COND_MARKER[kind_of(c["condition"])],
                   edgecolor="white", linewidth=0.6, alpha=0.9, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("mean policy calls per episode (log scale)")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.03, 1.0)
    ax.grid(color="0.9", lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    h1 = [Line2D([], [], marker="o", ls="", color=POLICY_STYLE[p][0], label=POLICY_LABEL[p], ms=7) for p in POLICY_ORDER]
    h2 = [Line2D([], [], marker=m, ls="", color="0.35", label=COND_LABEL[k], ms=7) for k, m in COND_MARKER.items()]
    leg1 = ax.legend(handles=h1, title="policy (colour)", fontsize=8.5, title_fontsize=8.5, frameon=False, loc="lower left",
                     bbox_to_anchor=(1.01, 0.52))
    ax.add_artist(leg1)
    ax.legend(handles=h2, title="perturbation (marker)", fontsize=8.5, title_fontsize=8.5, frameon=False, loc="lower left",
              bbox_to_anchor=(1.01, 0.05))
    ax.set_title("Cost vs success: one point per cell (40 cells, 200 episodes each)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/main")
    ap.add_argument("--out", default=None)
    ap.add_argument("--copy-to", default=None)
    a = ap.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    d = Path(a.dir) if Path(a.dir).is_absolute() else ROOT / a.dir
    out = Path(a.out) if a.out else d
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    cells = read_cells(d / "main_cells.csv")
    fig_main(cells, out / "fig_main.png")
    fig_cost(cells, out / "fig_cost.png")
    print(f"[make_main_figures] {len(cells)} cells -> {out / 'fig_main.png'}, {out / 'fig_cost.png'}")
    if a.copy_to:
        dst = Path(a.copy_to) if Path(a.copy_to).is_absolute() else ROOT / a.copy_to
        dst.mkdir(parents=True, exist_ok=True)
        for n in ("fig_main.png", "fig_cost.png", "fig_hypotheses.png"):
            shutil.copy2(out / n, dst / n)
        print(f"[make_main_figures] copied to {dst}")


if __name__ == "__main__":
    main()
