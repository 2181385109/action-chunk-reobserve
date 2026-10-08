"""T10b 自训练 ACT 正式分析：严格按 PREREGISTRATION_OWN.md 实现基础水平测试（描述）、H1'/H3' 的检验、次要指标描述、
探索性分析（差距变化、与预训练模型的成败一致率、成功率表）。

用法（WSL 内；run_main_own.sh 在基础水平测试 + 9 格全部完成后自动调用）：
  python scripts/analyze_main_own.py [--dir results/main_own] [--baseline-dir results/own_baseline] [--out results/main_own]
演示运行（用预训练模型 results/main/ 里对应的 9 格和 results/repro/act 当作“自训练模型”的数据；不是结果）：
  python scripts/analyze_main_own.py --dir results/main --baseline-dir results/repro/act --out results/main_own_dryrun --dry-run

输出（--out）：main_own_report.md、main_own_cells.csv（9 格成功率 / 次要指标）、hypotheses_own.json（H1'/H3'）、baseline_own.json、
  interaction_own.csv/json（差距变化）、concordance_own.csv/json（成败一致率）、fig_hypotheses_own.png、fig_success_own.png、fig_interaction_own.png。
DRY RUN 判定：加了 --dry-run，或者任何一格 / 基础水平测试的 config.json 里没有登记的检查点（路径 + model.safetensors 的 SHA-256）
  -> 报告开头标“DRY RUN，不是结果”。所以预训练模型的数据、或别的检查点的数据，不可能被当成正式结果。
所有回合都计入，不剔除；某格回合数 / 种子集合与登记的不一致时，该格标为不完整，涉及它的检验不做。
检验、bootstrap、Holm、“相反结果”与 PREREGISTRATION.md 相同（chunkre.stats）；Holm 族是 H1'、H3' 两个。
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

from analyze_interaction import compute_model as pretrained_gap_rows   # noqa: E402  （预训练模型的差距变化，复用）
from analyze_main import SECONDARY, STYLE, _fmt, _num, load_cell   # noqa: E402  （读一格 / 次要指标 / 格式化：与 ACT 分析共用）
from chunkre import main_own as mo  # noqa: E402
from chunkre.config import parse_seeds  # noqa: E402
from chunkre.metrics import wilson_ci  # noqa: E402
from chunkre.plotting import yerr_from_ci  # noqa: E402
from chunkre.stats import (ALPHA, BOOT_SEED, N_BOOT, bootstrap_mean_ci, classify, gap_change,  # noqa: E402
                           holm_adjust, mcnemar_exact, pair_counts, paired_bootstrap_diff)

POLICY_LABEL = {"k20": "k=20", "k50": "k=50", "k100": "k=100"}
NAN = float("nan")
REPRO_ACT = "results/repro/act"    # 预训练模型的基础水平复现（种子 1000–1499）


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------- 读数据
def load_grid(cell_dir: Path, seeds: list[int]) -> dict[str, dict]:
    return {c.name: load_cell(cell_dir, c.name, seeds) for c in mo.cells()}


def load_dir(directory: Path, seeds: list[int]) -> dict:
    """基础水平测试这类“目录本身就是一格”的结果。"""
    directory = Path(directory)
    return load_cell(directory.parent, directory.name, seeds)


def provenance(config_path: Path) -> tuple[bool, str]:
    """config.json 里记录的检查点是不是登记的那一个（路径 + model.safetensors 哈希）。返回 (是否一致, 说明)。"""
    if not Path(config_path).exists():
        return False, "no config.json"
    try:
        pol = json.loads(Path(config_path).read_text(encoding="utf-8")).get("policy", {})
    except Exception as e:   # noqa: BLE001
        return False, f"unreadable config.json ({e})"
    repo = pol.get("repo")
    sha = (pol.get("local_checkpoint_sha256") or {}).get("model.safetensors")
    if repo != mo.CHECKPOINT:
        return False, f"policy.repo = {repo!r}, not the registered checkpoint"
    if sha != mo.CHECKPOINT_SHA256:
        return False, f"model.safetensors SHA-256 = {sha!r}, not the registered one"
    return True, "registered checkpoint (path and SHA-256 match)"


def check_provenance(cell_dir: Path, baseline_dir: Path, data: dict, baseline: dict) -> list[dict]:
    """逐格核对检查点来源；有数据却不是登记检查点的，列出原因。"""
    out = []
    for c in mo.cells():
        if data[c.name]["rows"]:
            ok, why = provenance(Path(cell_dir) / c.name / "config.json")
            out.append(dict(item=c.name, ok=ok, why=why))
    if baseline["rows"]:
        ok, why = provenance(Path(baseline_dir) / "config.json")
        out.append(dict(item=mo.BASELINE_NAME, ok=ok, why=why))
    return out


# ---------------------------------------------------------------- 主要假设
def run_hypotheses(data: dict[str, dict]) -> list[dict]:
    """H1'/H3'：按种子配对的精确 McNemar（双侧）+ 成功率差与配对 bootstrap 区间；两个 p 值一起做 Holm 校正。"""
    res = []
    for hid, tag, A, B, acts in mo.HYPOTHESES:
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


# ---------------------------------------------------------------- 基础水平测试（只描述，不检验）
def baseline_summary(baseline: dict, repro: dict) -> dict:
    out = dict(official=mo.OFFICIAL_SUCCESS, weak_threshold=mo.WEAK_THRESHOLD, available=bool(baseline["rows"]),
               complete=baseline["complete"], issue=baseline["issue"])
    if baseline["rows"]:
        s = baseline["success"]
        lo, hi = wilson_ci(int(s.sum()), len(s))
        out.update(n=len(s), n_success=int(s.sum()), rate=float(s.mean()), wilson_lo=lo, wilson_hi=hi,
                   weak=bool(s.mean() < mo.WEAK_THRESHOLD))
    if repro["rows"]:
        s = repro["success"]
        lo, hi = wilson_ci(int(s.sum()), len(s))
        out.update(repro_n=len(s), repro_n_success=int(s.sum()), repro_rate=float(s.mean()), repro_wilson_lo=lo,
                   repro_wilson_hi=hi, repro_complete=repro["complete"])
    return out


# ---------------------------------------------------------------- 次要指标 / 推方块情况 / 最好的 k
def cell_table(data: dict[str, dict]) -> list[dict]:
    out = []
    for c in mo.cells():
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
    out = []
    for c in mo.cells():
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
    """探索性：每个条件下成功率最高的 k（并列全列出）和领先第二名多少个百分点。只描述，不检验。"""
    out = []
    for tag in mo.condition_tags():
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


# ---------------------------------------------------------------- 探索性：差距变化
def own_gap_rows(data: dict[str, dict]) -> list[dict]:
    """自训练模型的差距变化：k=20（短）对 k=100（长），条件推 0.06 / 延迟 10（定义同 PREREGISTRATION_DP.md §6.2）。"""
    S, L = mo.GAP["short"], mo.GAP["long"]
    rows = []
    for tag in mo.GAP["conds"]:
        r = dict(model="own", short=S, long=L, condition=tag)
        need = [data[f"{S}__{tag}"], data[f"{L}__{tag}"], data[f"{S}__none"], data[f"{L}__none"]]
        bad = [c for c in need if not c["complete"]]
        if bad:
            r.update(available=False, reason="; ".join(f"{c['name']}: {c['issue']}" for c in bad))
        else:
            r.update(available=True, **gap_change(*[c["success"] for c in need]))
        rows.append(r)
    return rows


def pre_gap_rows(pre_dir: Path, seeds: list[int]) -> list[dict]:
    """预训练模型同样条件的差距变化（复用 analyze_interaction.compute_model；只取推 0.06 / 延迟 10）。"""
    rows = [r for r in pretrained_gap_rows("ACT", pre_dir, seeds) if r["condition"] in mo.GAP["conds"]]
    for r in rows:
        r["model"] = "pretrained"
    return rows


# ---------------------------------------------------------------- 探索性：预训练 vs 自训练 在同一种子上的成败一致率
def concordance(own: dict, pre: dict, name: str) -> dict:
    r = dict(item=name)
    if not (own["complete"] and pre["complete"]):
        r.update(available=False, reason="; ".join(f"{c['name']}: {c['issue']}" for c in (own, pre) if not c["complete"]))
        return r
    if own["seeds"] != pre["seeds"]:
        r.update(available=False, reason="seed sets differ")
        return r
    pc = pair_counts(own["success"], pre["success"])   # a = 自训练，b = 预训练
    n = int(own["success"].size)
    r.update(available=True, n=n, both=pc["both"], only_own=pc["only_a"], only_pretrained=pc["only_b"], neither=pc["neither"],
             agreement=(pc["both"] + pc["neither"]) / n, rate_own=float(own["success"].mean()),
             rate_pretrained=float(pre["success"].mean()))
    return r


def all_concordance(data, pre_data, baseline, pre_baseline) -> list[dict]:
    rows = [concordance(data[c.name], pre_data[c.name], c.name) for c in mo.cells()]
    rows.append(concordance(baseline, pre_baseline, "baseline (k=100, seeds 1000–1499)"))
    return rows


# ---------------------------------------------------------------- 报告
def _pivot(table, key_fn):
    by = {t["cell"]: t for t in table}
    pols = [p for p, _, _ in mo.POLICIES]
    L = ["| condition | " + " | ".join(POLICY_LABEL[p] for p in pols) + " |", "|---|" + "---|" * len(pols)]
    for tag in mo.condition_tags():
        L.append(f"| {tag} | " + " | ".join(key_fn(by.get(f"{p}__{tag}")) for p in pols) + " |")
    return L


def build_report(ctx: dict) -> str:
    hyp, table, push, best, base = ctx["hyp"], ctx["table"], ctx["push"], ctx["best"], ctx["baseline"]
    L = []
    if ctx["dry_run"]:
        L += ["> **DRY RUN — NOT RESULTS.** " + ctx["dry_reason"] + " This report only checks that the analysis code runs. "
              "Nothing here is a finding about the self-trained model.", ""]
    L += ["# Self-trained ACT analysis report (T10b, implements PREREGISTRATION_OWN.md)", ""]
    L.append(f"- Registered checkpoint: `{mo.CHECKPOINT}` (model.safetensors SHA-256 `{mo.CHECKPOINT_SHA256[:16]}…`).")
    fails: dict[str, list[str]] = {}
    for p in ctx["provenance"]:
        if not p["ok"]:
            fails.setdefault(p["why"], []).append(p["item"])
    for why, items in fails.items():
        L.append(f"- **Checkpoint check failed** for {len(items)} result director{'y' if len(items) == 1 else 'ies'} "
                 f"({', '.join(items)}): {why}.")
    if ctx["provenance"] and all(p["ok"] for p in ctx["provenance"]):
        L.append(f"- Checkpoint check: all {len(ctx['provenance'])} result directories record the registered checkpoint "
                 "(path and SHA-256 in `config.json`).")
    L.append(f"- Cells: {sum(1 for t in table if t['complete'])}/{len(table)} complete "
             f"(expected {ctx['n']} episodes per cell, seeds {ctx['seeds_label']}, identical seed set in every cell).")
    L.append(f"- Tests: exact two-sided McNemar on seed-paired outcomes; paired bootstrap 95% CI "
             f"({N_BOOT} resamples, RNG seed {BOOT_SEED}); Holm correction over H1' and H3' only (alpha = {ALPHA}); "
             "separate from the ACT family H1–H3 and the DP family H4–H5.")
    for i in ctx["issues"]:
        L.append(f"- **INCOMPLETE** {i}")
    L.append("")

    L += ["## Base-level test (descriptive only, no test)", ""]
    if base["available"]:
        L += [f"Self-trained model, k = 100, no perturbation, seeds {ctx['baseline_seeds_label']}"
              + ("" if base["complete"] else f" (**INCOMPLETE**: {base['issue']})") + ":", "",
              "| model | successes | success rate | Wilson 95% CI |", "|---|---|---|---|",
              f"| self-trained (step 100000) | {base['n_success']}/{base['n']} | {base['rate']:.3f} | "
              f"[{base['wilson_lo']:.3f}, {base['wilson_hi']:.3f}] |"]
        if "repro_rate" in base:
            L.append(f"| pretrained hub model (our reproduction) | {base['repro_n_success']}/{base['repro_n']} | "
                     f"{base['repro_rate']:.3f} | [{base['repro_wilson_lo']:.3f}, {base['repro_wilson_hi']:.3f}] |")
        L.append(f"| official report | — | {base['official']:.3f} | — |")
        L.append("")
        if base["weak"]:
            L.append(f"**Weak-model rule triggered** (success rate {base['rate']:.3f} < {base['weak_threshold']:.2f}): the main "
                     "experiment stays as planned, and the README must state that *the self-trained model is weak; "
                     "the replication conclusions are of limited reference value*.")
        else:
            L.append(f"Weak-model rule (below {base['weak_threshold']:.2f}) not triggered.")
    else:
        L.append("No base-level test data found.")
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
              "A floor effect (almost no successes in both cells) shows up as a non-significant result and is not evidence "
              "against the ACT finding (PREREGISTRATION_OWN.md §5).", ""]

    L += ["## Success rate, all 9 cells (exploratory description; Wilson 95% CI)", ""]
    L += _pivot(table, lambda t: "—" if not t or "success_rate" not in t else
                f"{t['n_success']}/{t['n']} = {t['success_rate']:.3f} [{t['wilson_lo']:.3f}, {t['wilson_hi']:.3f}]"
                + ("" if t["complete"] else " (incomplete)"))
    L.append("")

    L += ["## Exploratory: gap change, self-trained vs. pretrained model (no test, not evidence)", "",
          "gap change = [success(k=20, perturbed) − success(k=100, perturbed)] − [success(k=20, none) − success(k=100, none)], "
          "averaged over the seeds, with a paired bootstrap 95% CI "
          f"({N_BOOT} resamples, RNG seed {BOOT_SEED}; same resampled seeds in all four cells). Positive = the short k "
          "loses less (or gains more) from the perturbation than the long k does. The pretrained model's values are computed "
          "from `" + ctx["pre_dir_label"] + "` on the same seeds.", "",
          "| model | condition | n | gap (perturbed) | gap (none) | **gap change** | 95% CI |", "|---|---|---|---|---|---|---|"]
    for r in ctx["gap_own"] + ctx["gap_pre"]:
        if not r["available"]:
            L.append(f"| {r['model']} | {r['condition']} | — | — | — | not computed: {r['reason']} | — |")
            continue
        L.append(f"| {r['model']} | {r['condition']} | {r['n']} | {r['gap_pert']:+.3f} | {r['gap_none']:+.3f} | "
                 f"**{r['gap_change']:+.3f}** | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] |")
    L.append("")

    L += ["## Exploratory: agreement of pretrained and self-trained model on the same seeds (descriptive only)", ""]
    if ctx["dry_run"]:
        L += ["> In this dry run both inputs are the same pretrained-model data, so the agreement is trivially 100% — it only "
              "checks that the code runs.", ""]
    L += ["Per seed, both models succeed / only the self-trained / only the pretrained / both fail; agreement = fraction of "
          "seeds with the same outcome. No test. (Outcomes are not noise-free under D3D12 rendering, so even a model compared "
          "with itself would not agree 100%.)", "",
          "| item | n | both succeed | only self-trained | only pretrained | both fail | agreement | rate self-trained | rate pretrained |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in ctx["conc"]:
        if not r["available"]:
            L.append(f"| {r['item']} | — | — | — | — | — | not computed: {r['reason']} | — | — |")
            continue
        L.append(f"| {r['item']} | {r['n']} | {r['both']} | {r['only_own']} | {r['only_pretrained']} | {r['neither']} | "
                 f"{r['agreement']:.3f} | {r['rate_own']:.3f} | {r['rate_pretrained']:.3f} |")
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

    L += ["## Best k per condition (EXPLORATORY; description only, no test)", "",
          "| condition | best k | success rate | runner-up | gap (pct. points) |", "|---|---|---|---|---|"]
    for b in best:
        L.append(f"| {b['condition']} | {', '.join(POLICY_LABEL[x] for x in b['best'])} | {b['rate']:.3f} | "
                 f"{POLICY_LABEL[b['runner_up']] if b['runner_up'] else '—'} | {_fmt(b['gap_pts'], 1)} |")
    L += ["", "Figures: `fig_hypotheses_own.png` (confirmatory), `fig_success_own.png` and `fig_interaction_own.png` (exploratory).", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- 图
def plot_figures(ctx: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = "DRY RUN (not results)" if ctx["dry_run"] else f"seeds {ctx['seeds_label']}"
    hyp, table = ctx["hyp"], ctx["table"]
    paths = []

    av = [r for r in hyp if r["available"]]
    fig, ax = plt.subplots(figsize=(6.4, 2.0 + 0.7 * max(1, len(av))))
    for i, r in enumerate(av):
        y = len(av) - 1 - i
        ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r["ci_lo"]], [r["ci_hi"] - r["diff"]]], fmt="o", color="#0072B2", capsize=3)
        ax.text(1.02, y, f"p_Holm = {_fmt(r.get('p_holm', NAN), 3)}", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(range(len(av)))
    ax.set_yticklabels([f"{r['id']}: {r['A']} − {r['B']}\n({r['condition']})" for r in reversed(av)], fontsize=8)
    ax.set_xlabel("difference in success rate (A − B), paired bootstrap 95% CI")
    ax.set_title(f"Primary hypotheses, self-trained ACT, {tag}", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    p = out_dir / "fig_hypotheses_own.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    by = {t["cell"]: t for t in table}
    tags = mo.condition_tags()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    nudge = np.linspace(-1, 1, len(mo.POLICIES)) * 0.08
    for i, (pol, _, _) in enumerate(mo.POLICIES):
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
    ax.set_title(f"EXPLORATORY: self-trained ACT, {tag}", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    p = out_dir / "fig_success_own.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    rows = [r for r in ctx["gap_own"] + ctx["gap_pre"] if r["available"]]
    fig, ax = plt.subplots(figsize=(7.0, 1.8 + 0.5 * max(1, len(rows))))
    color = {"own": "#0072B2", "pretrained": "#D55E00"}
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.errorbar(r["gap_change"], y, xerr=[[r["gap_change"] - r["ci_lo"]], [r["ci_hi"] - r["gap_change"]]], fmt="o",
                    color=color[r["model"]], capsize=3)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['model']}: k20 vs k100, {r['condition']}" for r in reversed(rows)], fontsize=8)
    ax.set_xlabel("gap change = (short−long, perturbed) − (short−long, none)", fontsize=9)
    ax.set_title(f"EXPLORATORY: change in the k=20 vs k=100 success gap\n({tag}; paired bootstrap 95% CI, no test)", fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    p = out_dir / "fig_interaction_own.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    return paths


def _csv_rows(rows, cols, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for t in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in t.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/main_own", help="含 <格子名>/episodes.csv 的目录（自训练模型的 9 格）")
    ap.add_argument("--baseline-dir", default="results/own_baseline", help="基础水平测试目录（自己就是一格：episodes.csv）")
    ap.add_argument("--out", default=None, help="输出目录（默认 = --dir）")
    ap.add_argument("--seeds", default=mo.OWN_SEEDS, help="9 格的预期种子")
    ap.add_argument("--baseline-seeds", default=mo.BASELINE_SEEDS)
    ap.add_argument("--pretrained-dir", default="results/main", help="预训练模型的对应 9 格（同样的格子名）")
    ap.add_argument("--pretrained-baseline-dir", default=REPRO_ACT, help="预训练模型的基础水平复现")
    ap.add_argument("--dry-run", action="store_true", help="强制标 DRY RUN（演示运行）")
    a = ap.parse_args(argv)
    d, bd, pre_d, pre_bd = _abs(a.dir), _abs(a.baseline_dir), _abs(a.pretrained_dir), _abs(a.pretrained_baseline_dir)
    out = _abs(a.out) if a.out else d
    out.mkdir(parents=True, exist_ok=True)
    seeds, bseeds = parse_seeds(a.seeds), parse_seeds(a.baseline_seeds)

    data = load_grid(d, seeds)
    baseline = load_dir(bd, bseeds)
    if not any(v["rows"] for v in data.values()) and not baseline["rows"]:
        raise SystemExit(f"{d} 下没有任何格子的 episodes.csv，{bd} 里也没有基础水平测试")
    pre_data, pre_baseline = load_grid(pre_d, seeds), load_dir(pre_bd, bseeds)

    prov = check_provenance(d, bd, data, baseline)
    bad = [p for p in prov if not p["ok"]]
    dry_run = a.dry_run or min(seeds) < 200000 or bool(bad) or not prov
    reasons = []
    if a.dry_run:
        reasons.append("run with --dry-run.")
    if min(seeds) < 200000:
        reasons.append("tuning seeds (< 200000) were used.")
    if not prov:
        reasons.append("no result directory with data.")
    if bad:
        reasons.append(f"{len(bad)} of {len(prov)} result directories do not record the registered self-trained checkpoint "
                       f"(e.g. {bad[0]['item']}: {bad[0]['why']}).")
    ctx = dict(dry_run=dry_run, dry_reason=" ".join(reasons), provenance=prov, n=len(seeds), seeds_label=a.seeds,
               baseline_seeds_label=a.baseline_seeds, pre_dir_label=str(a.pretrained_dir),
               issues=[f"{k}: {v['issue']}" for k, v in data.items() if not v["complete"]])
    ctx["hyp"] = run_hypotheses(data)
    ctx["table"] = cell_table(data)
    ctx["push"] = push_accounting(data)
    ctx["best"] = best_policies(ctx["table"])
    ctx["baseline"] = baseline_summary(baseline, pre_baseline)
    ctx["gap_own"] = own_gap_rows(data)
    ctx["gap_pre"] = pre_gap_rows(pre_d, seeds)
    ctx["conc"] = all_concordance(data, pre_data, baseline, pre_baseline)

    (out / "hypotheses_own.json").write_text(json.dumps(ctx["hyp"], indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    (out / "baseline_own.json").write_text(json.dumps(ctx["baseline"], indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    (out / "interaction_own.json").write_text(json.dumps(ctx["gap_own"] + ctx["gap_pre"], indent=2, ensure_ascii=False),
                                              encoding="utf-8", newline="\n")
    (out / "concordance_own.json").write_text(json.dumps(ctx["conc"], indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    cols = ["cell", "policy", "condition", "strength", "complete", "n", "n_success", "success_rate", "wilson_lo", "wilson_hi"]
    for col, _, _ in SECONDARY:
        cols += [f"{col}_mean", f"{col}_lo", f"{col}_hi", f"{col}_n"]
    _csv_rows(ctx["table"], cols, out / "main_own_cells.csv")
    _csv_rows(ctx["gap_own"] + ctx["gap_pre"], ["model", "short", "long", "condition", "available", "n", "rate_short_pert",
                                                "rate_long_pert", "gap_pert", "rate_short_none", "rate_long_none", "gap_none",
                                                "gap_change", "ci_lo", "ci_hi", "reason"], out / "interaction_own.csv")
    _csv_rows(ctx["conc"], ["item", "available", "n", "both", "only_own", "only_pretrained", "neither", "agreement",
                            "rate_own", "rate_pretrained", "reason"], out / "concordance_own.csv")
    figs = plot_figures(ctx, out)
    (out / "main_own_report.md").write_text(build_report(ctx), encoding="utf-8", newline="\n")

    b = ctx["baseline"]
    print(f"[analyze_main_own] {'DRY RUN ' if dry_run else ''}{sum(v['complete'] for v in data.values())}/{len(data)} 格完整 -> "
          f"{out / 'main_own_report.md'}、main_own_cells.csv、hypotheses_own.json、baseline_own.json、interaction_own.csv、"
          f"concordance_own.csv、{', '.join(f.name for f in figs)}")
    if b["available"]:
        print(f"  基础水平测试: {b['n_success']}/{b['n']} = {b['rate']:.3f} [{b['wilson_lo']:.3f}, {b['wilson_hi']:.3f}]"
              + ("；低于 40%：README 需注明“自训练模型较弱，重复结论的参考价值有限”" if b["weak"] else ""))
    for r in ctx["hyp"]:
        if r["available"]:
            print(f"  {r['id']} {r['condition']} {r['A']}−{r['B']}: diff {r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
                  f"p={r['p_raw']:.4g} p_Holm={_fmt(r.get('p_holm', NAN), 4)} {r.get('decision', '')}")
        else:
            print(f"  {r['id']}: not testable ({r['reason']})")
    if ctx["issues"]:
        print(f"[analyze_main_own] 不完整的格子 {len(ctx['issues'])} 个（见报告开头）")


if __name__ == "__main__":
    main()
