#!/usr/bin/env python3
"""README number check (T12).

Every number in the README's result tables must come from the files under `results/`, not from hand arithmetic.
This script reads the hypothesis json files and the cell csv files, formats the numbers exactly as the README
prints them, and checks that each README table row that should carry them really contains them (verbatim).
Any mismatch is an error (exit 1).

  python scripts/check_readme_numbers.py                 # check, write results/readme_check.txt
  python scripts/check_readme_numbers.py --print-tables  # print the README tables rendered from results/ (paste source)

Standard library only (runs with any Python 3.8+; no WSL needed).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
README = ROOT / "README.md"
OUT = R / "readme_check.txt"


# ------------------------------------------------------------------ loading / formatting
def jload(rel: str):
    return json.loads((R / rel).read_text(encoding="utf-8"))


def cload(rel: str) -> list[dict]:
    with open(R / rel, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _z(x: float) -> float:
    return 0.0 if abs(round(x, 3)) == 0 else x


def f3(x) -> str:
    return f"{_z(float(x)):.3f}"


def sg(x) -> str:
    x = _z(float(x))
    return f"{x:+.3f}"


def f1(x) -> str:
    return f"{float(x):.1f}"


def pct1(x) -> str:
    return f"{100 * float(x):.1f}%"


def ci(lo, hi) -> str:
    return f"[{sg(lo)}, {sg(hi)}]"


def wil(lo, hi) -> str:
    return f"[{float(lo):.3f}, {float(hi):.3f}]"


def fp(p) -> str:
    p = float(p)
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"{m}e{int(e)}"
    return f"{p:.2g}"


def cell_text(row: dict) -> str:
    return f"{int(row['n_success'])} {wil(row['wilson_lo'], row['wilson_hi'])}"


# ------------------------------------------------------------------ hypotheses
DECISION = {"supported": "supported", "not_significant": "**not supported** (not significant)", "opposite": "**opposite result**", "not supported": "**not supported** (not significant)"}


def hyp_sets() -> list[tuple[str, list[dict]]]:
    """(label, list of hypothesis dicts with a uniform key set)."""
    out = []
    for label, rel in [("ACT", "main/hypotheses.json"), ("DP", "main_dp/hypotheses_dp.json"), ("OWN", "main_own/hypotheses_own.json")]:
        out.append((label, jload(rel)))
    ad = jload("main_adaptive/hypotheses_adaptive.json")
    rows = []
    for k in ("HA1", "HA2"):
        h = dict(ad[k])
        h["p_raw"] = h["p_mcnemar"]
        h["p_holm"] = None
        rows.append(h)
    out.append(("ADAPTIVE", rows))
    return out


def hyp_row_checks() -> list[tuple[str, str, list[str]]]:
    rows = []
    for label, hs in hyp_sets():
        for h in hs:
            strings = [f3(h["rate_A"]), f3(h["rate_B"]), sg(h["diff"]), ci(h["ci_lo"], h["ci_hi"]),
                       f"{h['only_a']} / {h['only_b']}", fp(h["p_raw"])]
            if h.get("p_holm") is not None:
                strings.append(fp(h["p_holm"]))
            rows.append((f"hypothesis {h['id']}", f"| {h['id']} |", strings))
    return rows


def render_hyp_table(label: str) -> str:
    hs = dict(hyp_sets())[label]
    holm = label != "ADAPTIVE"
    head = "| H | Condition | A vs B | Success A | Success B | Diff (A − B) | Paired bootstrap 95% CI | Only A / only B | p (McNemar) |"
    head += " p (Holm) | Conclusion |" if holm else " Conclusion |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for h in hs:
        a, b = h["A"], h["B"]
        nice = lambda s: s.replace("k100", "k=100").replace("k20", "k=20").replace("k50", "k=50").replace("k15", "k=15") \
            .replace("k4", "k=4").replace("k8", "k=8").replace("adaptive", "adaptive").replace("te", "te")
        cond = h["condition"].replace("push0.06", "push 0.06 m").replace("push60", "push 60").replace("delay10", "delay 10") \
            .replace("delay4", "delay 4").replace("noise16", "noise σ=16").replace("none", "no perturbation")
        if label == "ADAPTIVE":
            concl = "supported" if h["id"] == "HA1" else "non-inferiority **not shown** (lower limit ≤ −0.05)"
            p = fp(h["p_raw"]) + ("" if h["id"] == "HA1" else " (descriptive)")
            lines.append(f"| {h['id']} | {cond} | {nice(a)} vs {nice(b)} | {f3(h['rate_A'])} | {f3(h['rate_B'])} | {sg(h['diff'])} | "
                         f"{ci(h['ci_lo'], h['ci_hi'])} | {h['only_a']} / {h['only_b']} | {p} | {concl} |")
        else:
            lines.append(f"| {h['id']} | {cond} | {nice(a)} vs {nice(b)} | {f3(h['rate_A'])} | {f3(h['rate_B'])} | {sg(h['diff'])} | "
                         f"{ci(h['ci_lo'], h['ci_hi'])} | {h['only_a']} / {h['only_b']} | {fp(h['p_raw'])} | {fp(h['p_holm'])} | "
                         f"{DECISION.get(h['decision'], h['decision'])} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ cell tables (success counts with Wilson intervals)
def cell_rows(rel: str) -> dict[tuple[str, str], dict]:
    return {(r["policy"], r["condition"]): r for r in cload(rel)}


def cell_table_checks() -> list[tuple[str, str, list[str]]]:
    rows = []
    for name, rel, pols, conds in [
        ("ACT cells", "main/main_cells.csv", ["k20", "k50", "k100", "te"],
         ["none", "push0.02", "push0.04", "push0.06", "noise4", "noise8", "noise16", "delay3", "delay5", "delay10"]),
        ("DP cells", "main_dp/main_dp_cells.csv", ["k4", "k8", "k15"], ["none", "push60", "delay2", "delay4"]),
        ("OWN cells", "main_own/main_own_cells.csv", ["k20", "k50", "k100"], ["none", "push0.06", "delay10"]),
    ]:
        cr = cell_rows(rel)
        for c in conds:
            rows.append((f"{name} {c}", f"| {c} |", [cell_text(cr[(p, c)]) for p in pols]))
    return rows


def render_cell_table(rel: str, pols: list[str], conds: list[str], head: list[str]) -> str:
    cr = cell_rows(rel)
    lines = ["| condition | " + " | ".join(head) + " |", "|---|" + "---|" * len(head)]
    for c in conds:
        lines.append(f"| {c} | " + " | ".join(cell_text(cr[(p, c)]) for p in pols) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ gap change
GAP_ROWS = [  # (display model, csv, csv model key, short, long)
    ("ACT (pretrained)", "main_dp/interaction/interaction.csv", "ACT"),
    ("ACT (self-trained)", "main_own/interaction_own.csv", "own"),
    ("DP", "main_dp/interaction/interaction.csv", "DP"),
]


def gap_rows() -> list[tuple[str, dict]]:
    out = []
    for disp, rel, key in GAP_ROWS:
        for r in cload(rel):
            if r["model"] == key:
                out.append((disp, r))
    return out


def gap_row_strings(r: dict) -> list[str]:
    return [f3(r["rate_short_pert"]), f3(r["rate_long_pert"]), sg(r["gap_pert"]), f3(r["rate_short_none"]), f3(r["rate_long_none"]),
            sg(r["gap_none"]), sg(r["gap_change"]), ci(r["ci_lo"], r["ci_hi"])]


def gap_checks() -> list[tuple[str, str, list[str]]]:
    rows = []
    for disp, r in gap_rows():
        prefix = f"| {disp} | {r['short'].replace('k', 'k=')} vs {r['long'].replace('k', 'k=')} | {r['condition']} |"
        rows.append((f"gap change {disp} {r['condition']}", prefix, gap_row_strings(r)))
    # the pretrained rows inside the self-trained file must equal the ACT rows (same data)
    act = {r["condition"]: r for d, r in gap_rows() if d == "ACT (pretrained)"}
    for r in cload("main_own/interaction_own.csv"):
        if r["model"] == "pretrained":
            a = act[r["condition"]]
            for k in ("gap_change", "ci_lo", "ci_hi", "gap_pert", "gap_none"):
                if abs(float(a[k]) - float(r[k])) > 1e-9:
                    rows.append((f"gap consistency {r['condition']} {k}", "<<consistency>>", [k]))
    return rows


def render_gap_table() -> str:
    head = ("| Model | Short vs long | Condition | Short, perturbed | Long, perturbed | Gap (perturbed) | Short, none | Long, none | "
            "Gap (none) | **Gap change** | Paired bootstrap 95% CI |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for disp, r in gap_rows():
        s = gap_row_strings(r)
        lines.append(f"| {disp} | {r['short'].replace('k', 'k=')} vs {r['long'].replace('k', 'k=')} | {r['condition']} | "
                     f"{s[0]} | {s[1]} | {s[2]} | {s[3]} | {s[4]} | {s[5]} | **{s[6]}** | {s[7]} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ self-trained: base level + concordance
def conc_rows() -> list[dict]:
    return cload("main_own/concordance_own.csv")


def conc_name(r: dict) -> str:
    return r["item"].replace("baseline (k=100, seeds 1000–1499)", "base level (k=100, seeds 1000–1499)")


def conc_strings(r: dict) -> list[str]:
    a = float(r["agreement"])
    return [r["both"], r["only_own"], r["only_pretrained"], r["neither"], f"{a:.3f}", f3(r["rate_own"]), f3(r["rate_pretrained"])]


def conc_checks() -> list[tuple[str, str, list[str]]]:
    return [(f"concordance {conc_name(r)}", f"| {conc_name(r)} |", conc_strings(r)) for r in conc_rows()]


def render_conc_table() -> str:
    head = "| Item | n | Both succeed | Only self-trained | Only pretrained | Both fail | Agreement | Rate self-trained | Rate pretrained |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in conc_rows():
        s = conc_strings(r)
        lines.append(f"| {conc_name(r)} | {r['n']} | {s[0]} | {s[1]} | {s[2]} | {s[3]} | {s[4]} | {s[5]} | {s[6]} |")
    return "\n".join(lines)


def baseline_strings() -> list[str]:
    b = jload("main_own/baseline_own.json")
    return [f"{b['n_success']}/{b['n']}", f3(b["rate"]), wil(b["wilson_lo"], b["wilson_hi"]),
            f"{b['repro_n_success']}/{b['repro_n']}", f3(b["repro_rate"]), wil(b["repro_wilson_lo"], b["repro_wilson_hi"]),
            f"{b['official']:.3f}", f"{b['weak_threshold']:.2f}"]


# ------------------------------------------------------------------ adaptive
def ad_rows() -> list[dict]:
    return cload("main_adaptive/main_adaptive_cells.csv")


POLNAME = {"k20": "k=20", "k50": "k=50", "k100": "k=100", "te": "te", "ad": "adaptive"}


def ad_strings(r: dict) -> list[str]:
    s = [f"{int(r['n_success'])}/{r['n']}", f3(r["rate"]), wil(r["wilson_lo"], r["wilson_hi"]), f1(r["mean_calls"])]
    if r["paired"] == "True":
        s.append(f"{sg(r['diff_vs_adaptive'])} {ci(r['diff_lo'], r['diff_hi'])}")
    return s


def ad_checks() -> list[tuple[str, str, list[str]]]:
    rows = []
    for r in ad_rows():
        rows.append((f"adaptive descriptive {r['condition']} {r['policy']}", f"| {r['condition']} / {POLNAME[r['policy']]} |", ad_strings(r)))
    return rows


def render_ad_table() -> str:
    head = "| Condition / policy | Successes | Rate | Wilson 95% CI | Mean policy calls | Adaptive − policy (paired bootstrap 95% CI) |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in ad_rows():
        s = ad_strings(r)
        last = s[4] if len(s) > 4 else "—"
        lines.append(f"| {r['condition']} / {POLNAME[r['policy']]} | {s[0]} | {s[1]} | {s[2]} | {s[3]} | {last} |")
    return "\n".join(lines)


def ad_ratio(cond: str, base: str = "k50") -> float:
    d = {(r["condition"], r["policy"]): r for r in ad_rows()}
    return float(d[(cond, "ad")]["mean_calls"]) / float(d[(cond, base)]["mean_calls"])


def prose_checks() -> list[tuple[str, str]]:
    """(label, exact README substring) for numbers that appear in running text."""
    ad = jload("main_adaptive/hypotheses_adaptive.json")
    rc = ad["reaction"]
    rep = {r["condition"]: r for r in ad["replans"]}
    sel = jload("tune_adaptive/selection.json")
    rows = {r["combo"]: r for r in sel["rows"]}
    chosen = rows[sel["selected"]]
    others = [r["push_success"] for r in sel["rows"] if r["eligible"] and r["combo"] != sel["selected"]]
    adc = {(r["condition"], r["policy"]): r for r in ad_rows()}
    ad_push = adc[("push0.06", "ad")]
    ad_none = adc[("none", "ad")]
    own_b = jload("main_own/baseline_own.json")
    main_cells = cell_rows("main/main_cells.csv")
    mc = lambda p, c: f3(main_cells[(p, c)]["success_rate"])
    out = [
        ("self-trained base level", f"{own_b['n_success']}/{own_b['n']} = {f3(own_b['rate'])}"),
        ("self-trained base level (headline)", pct1(own_b["rate"])),
        ("adaptive calls ratio push", f"≈{round(ad_ratio('push0.06')):d}×"),
        ("adaptive calls none mean", f1(ad_none["mean_calls"])),
        ("adaptive calls push mean", f1(ad_push["mean_calls"])),
        ("k50 calls none mean", f1(adc[("none", "k50")]["mean_calls"])),
        ("k50 calls push mean", f1(adc[("push0.06", "k50")]["mean_calls"])),
        ("tuning push (selected)", f"{chosen['push_success']}/{chosen['push_n']}"),
        ("tuning push (selected) pct", pct1(chosen["push_rate"])),
        ("tuning other eligible range", f"{min(others)}–{max(others)}/{sel['n_per_cell']}"),
        ("formal push", f"{int(ad_push['n_success'])}/{ad_push['n']}"),
        ("formal push pct", pct1(ad_push["rate"])),
        ("tuning combos", f"{len(sel['rows'])} combinations"),
        ("tuning episodes", f"{sel['n_per_cell']} episodes"),
        ("reaction: median", f"median {rc['median']:.0f}"),
        ("reaction: Q1", f"Q1 {rc['q1']:.0f}"),
        ("reaction: Q3", f"Q3 {rc['q3']:.1f}"),
        ("reaction: range", f"{rc['min']}–{rc['max']}"),
        ("reaction: within 10", pct1(rc["frac_within"]["10"])),
        ("reaction: within 20", pct1(rc["frac_within"]["20"])),
        ("reaction: within 30", pct1(rc["frac_within"]["30"])),
        ("reaction: n with reaction", f"{rc['n_with_reaction']} of {rc['n_pushed']}"),
        ("reaction: replan exactly at push step", f"{rc['n_replan_at_push_step']} episodes"),
        ("replans none mean", f"{rep['none']['mean_diff_replans']:.2f}"),
        ("replans push mean", f"{rep['push0.06']['mean_diff_replans']:.2f}"),
        ("replans none share", pct1(rep["none"]["frac_with_diff_replan"])),
        ("replans push share", pct1(rep["push0.06"]["frac_with_diff_replan"])),
        # ACT descriptive bullet (kept from the first README version)
        ("ACT none k100", mc("k100", "none")), ("ACT none k20", mc("k20", "none")),
        ("ACT push0.06 k20", mc("k20", "push0.06")), ("ACT push0.06 k100", mc("k100", "push0.06")),
        ("ACT delay5 te", mc("te", "delay5")), ("ACT delay10 te", mc("te", "delay10")),
        ("ACT delay5 k20", mc("k20", "delay5")), ("ACT delay10 k20", mc("k20", "delay10")),
    ]
    ag = [float(r["agreement"]) for r in conc_rows() if r["item"].startswith("k") and float(r["agreement"]) < 1.0]
    out.append(("agreement range (same-seed, cells with a success)", f"{min(ag):.3f}–{max(ag):.3f}"))
    # DP baseline-gap sentence: short-vs-long gap without perturbation and under the perturbation
    for disp, r in gap_rows():
        if disp == "DP" and r["condition"] == "push60":
            out.append(("DP gap none", sg(r["gap_none"])))
            out.append(("DP gap push", sg(r["gap_pert"])))
    return out


# ------------------------------------------------------------------ main
def all_row_checks() -> list[tuple[str, str, list[str]]]:
    return hyp_row_checks() + cell_table_checks() + gap_checks() + conc_checks() + ad_checks()


def run_checks() -> tuple[list[str], list[str]]:
    text = README.read_text(encoding="utf-8")
    lines = text.splitlines()
    ok, bad = [], []
    for label, prefix, strings in all_row_checks():
        if prefix == "<<consistency>>":
            bad.append(f"{label}: pretrained rows differ between interaction.csv and interaction_own.csv")
            continue
        cand = [ln for ln in lines if ln.startswith(prefix)]
        if not cand:
            bad.append(f"{label}: no README line starts with {prefix!r}")
            continue
        miss = None
        for ln in cand:
            m = [s for s in strings if s not in ln]
            if not m:
                miss = []
                break
            if miss is None or len(m) < len(miss):
                miss = m
        if miss:
            bad.append(f"{label}: README row {prefix!r} lacks {miss}")
        else:
            ok.append(f"{label}: {len(strings)} numbers ok")
    for label, s in prose_checks():
        (ok if s in text else bad).append(f"prose {label}: {s!r}" + ("" if s in text else " NOT FOUND in README"))
    for s in baseline_strings():
        (ok if s in text else bad).append(f"baseline {s!r}" + ("" if s in text else " NOT FOUND in README"))
    return ok, bad


def main() -> int:
    if "--print-tables" in sys.argv:
        print("## hyp ACT\n" + render_hyp_table("ACT"))
        print("\n## hyp DP\n" + render_hyp_table("DP"))
        print("\n## hyp OWN\n" + render_hyp_table("OWN"))
        print("\n## hyp ADAPTIVE\n" + render_hyp_table("ADAPTIVE"))
        print("\n## cells DP\n" + render_cell_table("main_dp/main_dp_cells.csv", ["k4", "k8", "k15"], ["none", "push60", "delay2", "delay4"], ["k=4", "k=8", "k=15"]))
        print("\n## cells OWN\n" + render_cell_table("main_own/main_own_cells.csv", ["k20", "k50", "k100"], ["none", "push0.06", "delay10"], ["k=20", "k=50", "k=100"]))
        print("\n## gap\n" + render_gap_table())
        print("\n## concordance\n" + render_conc_table())
        print("\n## adaptive descriptive\n" + render_ad_table())
        print("\n## baseline strings\n" + " | ".join(baseline_strings()))
        print("\n## prose\n" + "\n".join(f"{a}: {b}" for a, b in prose_checks()))
        return 0
    ok, bad = run_checks()
    lines = [f"README number check: {len(ok)} ok, {len(bad)} problems", f"README: {README.name}; sources: results/**/*.json, *.csv", ""]
    lines += [f"FAIL  {b}" for b in bad] + [f"ok    {o}" for o in ok]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:2 + len(bad) + 1]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
