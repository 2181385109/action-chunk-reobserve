"""T6a 试跑汇总：各格 episodes.csv -> pilot_table.md / pilot_table.csv / fig_*.png。只描述，不做检验，不下结论。

用法（WSL 内；run_pilot.sh 结束时自动调用）：
  python scripts/analyze_pilot.py [--dir results/pilot] [--expected-n 20]
每格：成功率（Wilson 95% 区间）、平均策略调用次数、平均 path_length、平均 jerk（都对该格全部回合取平均，
成功和失败的回合混在一起；path_length 是整回合的总和，所以回合越长越大）。
成功率 <= 10% 标“地板”，>= 90% 标“天花板”（只是机械地按阈值标，n=20 时 10% = 2 回合）。
图：每种扰动一张，横轴强度（0 = 无扰动），纵轴成功率，每个策略一条线，竖线 = Wilson 95% 区间。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunkre.metrics import read_episodes, wilson_ci  # noqa: E402
from chunkre.pilot import CEIL, CSV_COLUMNS, FLOOR, PERTURBS, POLICIES, cells, flag_of  # noqa: E402

FLAG_MARK = {"floor": " ▼floor", "ceiling": " ▲ceil", "": ""}
NAN = float("nan")


def _mean(rows, col):
    v = [float(r[col]) for r in rows if r.get(col) not in ("", None)]
    return float(np.nanmean(v)) if v and not np.all(np.isnan(v)) else NAN


def cell_stats(cell_dir: Path, c) -> dict | None:
    """一格的统计；目录或 CSV 不存在返回 None。"""
    p = Path(cell_dir) / c.name / "episodes.csv"
    if not p.exists():
        return None
    rows = read_episodes(p)
    n = len(rows)
    ns = int(sum(int(float(r["success"])) for r in rows))
    lo, hi = wilson_ci(ns, n)
    rate = ns / n if n else NAN
    return dict(cell=c.name, policy=c.policy, perturb=c.perturb_tag, strength=c.strength, n=n, n_success=ns,
                success_rate=rate, ci_lo=lo, ci_hi=hi,
                mean_policy_calls=_mean(rows, "n_policy_calls"), mean_path_length=_mean(rows, "path_length"),
                mean_jerk=_mean(rows, "jerk"), mean_steps=_mean(rows, "steps"), flag=flag_of(rate))


def collect(cell_dir: Path) -> tuple[list[dict], list[str]]:
    stats, missing = [], []
    for c in cells():
        s = cell_stats(cell_dir, c)
        if s is None:
            missing.append(c.name)
        else:
            stats.append(s)
    return stats, missing


def _fmt(x, nd=3):
    return "—" if x != x else f"{x:.{nd}f}"


def write_csv(stats, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for s in stats:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in s.items()})


def build_markdown(stats, missing, expected_n: int) -> str:
    by = {s["cell"]: s for s in stats}
    L = ["# T6a 试跑结果（调参种子，只用于定档位，不作为结论）", ""]
    L.append(f"- 格子：找到 {len(stats)}/{len(cells())}；每格预期 {expected_n} 回合。"
             f"成功率 <= {FLOOR:.0%} 标 ▼floor（地板），>= {CEIL:.0%} 标 ▲ceil（天花板）。")
    partial = [s["cell"] for s in stats if s["n"] != expected_n]
    if missing:
        L.append(f"- 缺格（未跑 / 没有 episodes.csv）：{', '.join(missing)}")
    if partial:
        names = [f"{c}（{by[c]['n']}）" for c in partial]
        L.append("- 回合数不足的格子：" + ", ".join(names))
    L.append("- 成功率括号内是 Wilson 95% 区间；平均值对该格全部回合取（成功和失败混合），path_length 是整回合总和。")
    L.append("")

    # 表 1：成功率透视（行 = 条件，列 = 策略）
    pol = [p for p, _, _ in POLICIES]
    conds = []
    for short, kind, levels in PERTURBS:
        for s in levels:
            conds.append(("none" if kind == "none" else f"{short}{s:g}"))
    L += ["## 成功率（成功数/回合数；Wilson 95% 区间）", "", "| 条件 | " + " | ".join(pol) + " |", "|---|" + "---|" * len(pol)]
    for cnd in conds:
        cellsx = []
        for p in pol:
            s = by.get(f"{p}__{cnd}")
            if s is None:
                cellsx.append("—")
            else:
                cellsx.append(f"{s['n_success']}/{s['n']} {s['success_rate']:.2f} "
                              f"[{s['ci_lo']:.2f}, {s['ci_hi']:.2f}]{FLAG_MARK[s['flag']]}")
        L.append(f"| {cnd} | " + " | ".join(cellsx) + " |")
    L.append("")

    # 表 2–4：另外三个指标的透视
    for title, key, nd in [("平均策略调用次数", "mean_policy_calls", 1), ("平均 path_length", "mean_path_length", 2),
                           ("平均 jerk", "mean_jerk", 4)]:
        L += [f"## {title}", "", "| 条件 | " + " | ".join(pol) + " |", "|---|" + "---|" * len(pol)]
        for cnd in conds:
            L.append(f"| {cnd} | " + " | ".join(
                _fmt(by[f"{p}__{cnd}"][key], nd) if f"{p}__{cnd}" in by else "—" for p in pol) + " |")
        L.append("")

    fl = [s for s in stats if s["flag"]]
    L += ["## 地板 / 天花板格子（机械标注）", ""]
    if fl:
        L += ["| 格子 | 成功数/回合数 | 标记 |", "|---|---|---|"]
        L += [f"| {s['cell']} | {s['n_success']}/{s['n']} | {s['flag']} |" for s in fl]
    else:
        L.append("无")
    L += ["", "完整逐格数据见 `pilot_table.csv`；图：`fig_push.png`、`fig_noise.png`、`fig_delay.png`。", ""]
    return "\n".join(L)


def plot_figures(stats, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {s["cell"]: s for s in stats}
    # Okabe-Ito 色盲友好；时间集成用黑色虚线
    style = {"k10": ("#0072B2", "o", "-"), "k20": ("#009E73", "s", "-"), "k50": ("#E69F00", "^", "-"),
             "k100": ("#D55E00", "D", "-"), "te": ("#000000", "v", "--")}
    label = {"k10": "k=10", "k20": "k=20", "k50": "k=50", "k100": "k=100", "te": "temporal ensemble"}
    xlabel = {"push": "push distance d (m)", "noise": "image noise sigma (0-255)", "delay": "action delay d (steps)"}
    paths = []
    for short, kind, levels in PERTURBS:
        if kind == "none":
            continue
        xs = [0.0] + [float(s) for s in levels]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.axhspan(0, FLOOR, color="0.88", zorder=0)
        ax.axhspan(CEIL, 1, color="0.88", zorder=0)
        ax.axhline(FLOOR, color="0.6", lw=0.8, ls=":")
        ax.axhline(CEIL, color="0.6", lw=0.8, ls=":")
        nudge = np.linspace(-1, 1, len(POLICIES)) * 0.012 * (xs[-1] - xs[0])   # 横向错开一点，避免区间线重叠
        for i, (pol, _, _) in enumerate(POLICIES):
            col, mk, ls = style[pol]
            pts = []
            for x, s in zip(xs, [by.get(f"{pol}__none")] + [by.get(f"{pol}__{short}{lv:g}") for lv in levels]):
                if s is not None:
                    pts.append((x + nudge[i], s["success_rate"], s["ci_lo"], s["ci_hi"], s["flag"]))
            if not pts:
                continue
            px, py, lo, hi, fl = (np.array([p[j] for p in pts]) for j in range(5))
            ax.errorbar(px, py, yerr=[py - lo, hi - py], color=col, marker=mk, ls=ls, lw=1.6, ms=5, capsize=2,
                        elinewidth=0.9, alpha=0.95, label=label[pol], zorder=3)
            hit = fl != ""
            if hit.any():   # 地板 / 天花板点：空心圈标出
                ax.scatter(px[hit], py[hit], s=95, facecolors="none", edgecolors=col, linewidths=1.4, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs])
        ax.set_xlabel(xlabel[short] + "  (0 = no perturbation)")
        ax.set_ylabel("success rate (Wilson 95% CI)")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"T6a pilot, aloha ACT, seeds 5000-5019: {short}", fontsize=10)
        fig.text(0.01, 0.01, "gray bands: floor (success <= 10%) and ceiling (>= 90%); open circles = cells in a band",
                 fontsize=7, color="0.4")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        p = out_dir / f"fig_{short}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/pilot", help="试跑输出目录（含 <格子名>/episodes.csv）")
    ap.add_argument("--expected-n", type=int, default=20, dest="expected_n", help="每格预期回合数（默认 20）")
    a = ap.parse_args(argv)
    d = Path(a.dir)
    if not d.is_absolute():
        d = ROOT / d
    stats, missing = collect(d)
    if not stats:
        raise SystemExit(f"{d} 下没有任何格子的 episodes.csv")
    write_csv(stats, d / "pilot_table.csv")
    (d / "pilot_table.md").write_text(build_markdown(stats, missing, a.expected_n), encoding="utf-8", newline="\n")
    figs = plot_figures(stats, d)
    print(f"[analyze_pilot] {len(stats)}/{len(cells())} 格 -> {d / 'pilot_table.md'}、pilot_table.csv、{', '.join(f.name for f in figs)}")
    if missing:
        print(f"[analyze_pilot] 缺 {len(missing)} 格：{', '.join(missing[:6])}{' …' if len(missing) > 6 else ''}")


if __name__ == "__main__":
    main()
