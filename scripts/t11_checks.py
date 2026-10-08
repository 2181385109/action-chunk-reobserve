"""T11a 自适应重新观察规则检查（结果 results/t11_checks/；一键运行 scripts/t11_checks.sh，最后调用本脚本 analyze）。

A1 退化检查：tau=inf（永远不触发）时 = k=100。CPU 渲染，种子 0–9，B=10，--log-traj：a1_k100 vs a1_adaptive_inf。
   比较除策略调用次数、墙钟列、mode 列和 adaptive 列以外的所有列，以及整回合实际动作（轨迹 action，逐位）。
   另外与 T5 的 results/t5_checks/n1_act_none（T5 时的代码，CPU，k=100，同种子）比较，确认 fixed 模式没被改动。
A2 tau=0、min_gap=m：换计划间隔恰好为 m（m=10、20；GPU，种子 0–9）。用 adaptive/*.npz 的换计划记录和
   轨迹确认：换计划那一步起执行的动作 = 那一步预测的候选块（轨迹 chunk_sel，块内第 1/2/5/10… 个动作，逐位）。
A3 差异分布：GPU，种子 0–19，m=10，tau=inf，无扰动 / 推 0.06：所有比较的差异值分位数（50/75/90/95/99%），
   以及推的那一步之后第一次比较的差异与无扰动同一种子同一步的差异对比（配对）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunkre.adaptive import ADAPTIVE_COLUMNS, CAUSE_DIFF, load_checks  # noqa: E402
from chunkre.metrics import TIME_COLUMNS, read_episodes  # noqa: E402

OUT = ROOT / "results" / "t11_checks"
QUANTILES = (50, 75, 90, 95, 99)
TRAJ_SAME_KEYS = ("seeds", "steps", "alive", "action", "agent_pos", "cube_pos")


def _load_traj(run: Path) -> list:
    return [dict(np.load(p)) for p in sorted((run / "traj").glob("batch_*.npz"))]


def _load_adapt(run: Path) -> list:
    return [dict(np.load(p)) for p in sorted((run / "adaptive").glob("batch_*.npz"))]


def compare_runs(a: Path, b: Path) -> dict:
    """逐回合 CSV（排除调用次数 / 墙钟 / mode / adaptive 列）+ 轨迹数组逐位比较。"""
    ra, rb = read_episodes(a / "episodes.csv"), read_episodes(b / "episodes.csv")
    skip = set(TIME_COLUMNS) | {"n_policy_calls", "mode"} | set(ADAPTIVE_COLUMNS)
    cols = [c for c in ra[0] if c not in skip and c in rb[0]]
    key = lambda r: int(r["seed"])  # noqa: E731
    ra, rb = sorted(ra, key=key), sorted(rb, key=key)
    assert [key(r) for r in ra] == [key(r) for r in rb]
    same_ep = [all(x[c] == y[c] for c in cols) for x, y in zip(ra, rb)]
    diff_cols = sorted({c for x, y in zip(ra, rb) for c in cols if x[c] != y[c]})
    ta, tb = _load_traj(a), _load_traj(b)
    acts_same, keys_same = [], {}
    for za, zb in zip(ta, tb):
        for i in range(len(za["seeds"])):
            n = int(za["steps"][i])
            acts_same.append(bool(int(zb["steps"][i]) == n and np.array_equal(za["action"][:n, i], zb["action"][:n, i])))
        for k in TRAJ_SAME_KEYS:
            if k in za:
                keys_same[k] = keys_same.get(k, True) and k in zb and za[k].shape == zb[k].shape and \
                    bool(np.array_equal(za[k], zb[k], equal_nan=za[k].dtype.kind == "f"))
    return dict(a=str(a.relative_to(ROOT)), b=str(b.relative_to(ROOT)), n=len(ra), compared_columns=cols,
                episodes_identical=int(sum(same_ep)), differing_columns=diff_cols,
                full_episode_actions_identical=int(sum(acts_same)),
                traj_arrays_identical={k: bool(v) for k, v in keys_same.items()},
                success_a=int(sum(int(r["success"]) for r in ra)), success_b=int(sum(int(r["success"]) for r in rb)),
                mean_calls_a=float(np.mean([int(r["n_policy_calls"]) for r in ra])),
                mean_calls_b=float(np.mean([int(r["n_policy_calls"]) for r in rb])))


def a1() -> dict:
    out = dict(adaptive_vs_k100=compare_runs(OUT / "a1_k100", OUT / "a1_adaptive_inf"))
    t5 = ROOT / "results" / "t5_checks" / "n1_act_none"
    if (t5 / "episodes.csv").exists():
        out["k100_vs_t5_reference"] = compare_runs(t5, OUT / "a1_k100")
    rows = read_episodes(OUT / "a1_adaptive_inf" / "episodes.csv")
    out["adaptive_replans_diff_total"] = int(sum(int(r["n_replans_diff"]) for r in rows))
    c = out["adaptive_vs_k100"]
    out["pass"] = bool(c["episodes_identical"] == c["n"] == c["full_episode_actions_identical"]
                       and out["adaptive_replans_diff_total"] == 0)
    return out


def _a2_run(run: Path, m: int) -> dict:
    ad, tr = _load_adapt(run), _load_traj(run)
    rows = {int(r["seed"]): r for r in read_episodes(run / "episodes.csv")}
    n_ep = n_int = n_bad_int = n_bad_cause = n_chk_act = n_bad_act = n_bad_calls = n_bad_planj = 0
    intervals = []
    for z, t in zip(ad, tr):
        offs = [int(o) for o in t["chunk_offsets"]]
        for i, seed in enumerate(z["seeds"]):
            n_ep += 1
            sel = z["rp_env"] == i
            ts, causes = z["rp_t"][sel].astype(int), z["rp_cause"][sel].astype(int)
            iv = np.diff(ts)
            intervals += iv.tolist()
            n_int += len(iv)
            n_bad_int += int((iv != m).sum())
            n_bad_cause += int((causes[1:] != CAUSE_DIFF).sum())
            # 换计划那一步起执行的是那一步的候选块：action[t+o] == chunk_sel[t, o]（o < m，回合内）
            steps = int(z["steps"][i])
            for t0 in ts:
                for jo, o in enumerate(offs):
                    if o < m and t0 + o < steps:
                        n_chk_act += 1
                        n_bad_act += int(not np.array_equal(t["action"][t0 + o, i], t["chunk_sel"][t0, i, jo]))
            pj = z["plan_j"][:steps, i].astype(int)
            n_bad_planj += int(not np.array_equal(pj, np.arange(steps) % m))
            r = rows[int(seed)]
            n_bad_calls += int(int(r["n_policy_calls"]) != 1 + int(r["n_replans_diff"])
                               or int(r["n_replans_exhaust"]) != 0)
    return dict(run=str(run.relative_to(ROOT)), m=m, episodes=n_ep, intervals=n_int, intervals_not_m=n_bad_int,
                interval_values=sorted(set(intervals)), non_diff_replans=n_bad_cause,
                action_vs_candidate_checked=n_chk_act, action_vs_candidate_mismatch=n_bad_act,
                plan_j_not_t_mod_m=n_bad_planj, calls_not_1_plus_replans=n_bad_calls,
                passed=bool(n_bad_int == 0 and n_int > 0 and n_bad_cause == 0 and n_bad_act == 0 and n_chk_act > 0
                            and n_bad_planj == 0 and n_bad_calls == 0))


def a2() -> dict:
    out = {f"m{m}": _a2_run(OUT / f"a2_m{m}", m) for m in (10, 20)}
    out["pass"] = all(v["passed"] for v in out.values())
    return out


def quantiles(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {f"p{q}": float(np.percentile(x, q)) for q in QUANTILES} | dict(n=int(len(x)), mean=float(x.mean()),
                                                                           min=float(x.min()), max=float(x.max()))


def a3() -> dict:
    none, push = load_checks(OUT / "a3_none"), load_checks(OUT / "a3_push0.06")
    eps = {int(r["seed"]): r for r in read_episodes(OUT / "a3_push0.06" / "episodes.csv")}
    none_at = {(c["seed"], c["t"]): c["diff"] for c in none}
    out = dict(none=quantiles([c["diff"] for c in none]), push0_06=quantiles([c["diff"] for c in push]))
    # 推的那一步之后第一次比较（t > push_step），与无扰动同种子同一步配对
    first_after, paired, pre = [], [], []
    for seed, r in sorted(eps.items()):
        if r.get("push_status") != "pushed":
            continue
        p = int(r["push_step"])
        cs = sorted((c for c in push if c["seed"] == seed), key=lambda c: c["t"])
        pre += [c["diff"] for c in cs if c["t"] <= p]
        after = [c for c in cs if c["t"] > p]
        if not after:
            continue
        c = after[0]
        first_after.append(c["diff"])
        if (seed, c["t"]) in none_at:
            paired.append(dict(seed=seed, push_step=p, t=c["t"], push=c["diff"], none=none_at[(seed, c["t"])]))
    ratio = [d["push"] / d["none"] for d in paired if d["none"] > 0]
    pn = np.array([[d["push"], d["none"]] for d in paired]) if paired else np.zeros((0, 2))
    out["after_push"] = dict(
        n_pushed=sum(1 for r in eps.values() if r.get("push_status") == "pushed"),
        push_status_counts={s: sum(1 for r in eps.values() if r.get("push_status") == s)
                            for s in sorted({r.get("push_status") for r in eps.values()})},
        first_check_after_push=quantiles(first_after) if first_after else None,
        checks_before_push=quantiles(pre) if pre else None,
        n_paired=len(paired), paired=paired,
        paired_push_gt_none=int((pn[:, 0] > pn[:, 1]).sum()) if len(pn) else 0,
        paired_median_push=float(np.median(pn[:, 0])) if len(pn) else None,
        paired_median_none=float(np.median(pn[:, 1])) if len(pn) else None,
        paired_median_ratio=float(np.median(ratio)) if ratio else None,
        first_after_above_none_p95=int(sum(d > out["none"]["p95"] for d in first_after)),
        first_after_above_none_p99=int(sum(d > out["none"]["p99"] for d in first_after)),
    )
    # 定 tau 候选值：无扰动分布的 50/75/90/95/99% 分位数（四舍五入到 4 位小数，T11a 写进 src/chunkre/tune_adaptive.py）
    out["tau_candidates"] = [round(out["none"][f"p{q}"], 4) for q in QUANTILES]
    return out


def _fmt(q: dict | None) -> str:
    if not q:
        return "—"
    return " / ".join(f"{q[f'p{x}']:.4f}" for x in QUANTILES) + f"（n={q['n']}，均值 {q['mean']:.4f}，最大 {q['max']:.4f}）"


def analyze() -> None:
    res = dict(a1=a1(), a2=a2(), a3=a3())
    (OUT / "checks.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    c = res["a1"]["adaptive_vs_k100"]
    lines = ["# T11a 自适应规则检查（scripts/t11_checks.sh 生成）", "",
             f"## A1 tau=inf vs k=100（CPU 渲染，种子 0–9）：{'通过' if res['a1']['pass'] else '未通过'}",
             f"- 逐回合一致 {c['episodes_identical']}/{c['n']}（比较列 {len(c['compared_columns'])} 个：除 n_policy_calls、"
             f"墙钟、mode、adaptive 列外全部），整回合动作逐位相同 {c['full_episode_actions_identical']}/{c['n']}；"
             f"轨迹数组 {c['traj_arrays_identical']}；差异触发换计划 {res['a1']['adaptive_replans_diff_total']} 次",
             f"- 成功 {c['success_a']} vs {c['success_b']}；平均策略调用 k=100 {c['mean_calls_a']:.1f} vs adaptive {c['mean_calls_b']:.1f}"]
    if "k100_vs_t5_reference" in res["a1"]:
        t = res["a1"]["k100_vs_t5_reference"]
        lines.append(f"- 本次 k=100 与 T5 参照（results/t5_checks/n1_act_none）：逐回合一致 {t['episodes_identical']}/{t['n']}，"
                     f"动作逐位相同 {t['full_episode_actions_identical']}/{t['n']}（fixed 模式未被改动）")
    lines += ["", f"## A2 tau=0、min_gap=m：{'通过' if res['a2']['pass'] else '未通过'}"]
    for k in ("m10", "m20"):
        v = res["a2"][k]
        lines.append(f"- m={v['m']}：{v['episodes']} 回合，换计划间隔 {v['intervals']} 个，取值 {v['interval_values']}，"
                     f"≠m 的 {v['intervals_not_m']} 个；非差异触发 {v['non_diff_replans']}；执行动作 = 候选块 "
                     f"{v['action_vs_candidate_checked'] - v['action_vs_candidate_mismatch']}/{v['action_vs_candidate_checked']}；"
                     f"plan_j ≠ t mod m 的回合 {v['plan_j_not_t_mod_m']}；调用 ≠ 1+换计划 {v['calls_not_1_plus_replans']}")
    a = res["a3"]
    ap = a["after_push"]
    lines += ["", "## A3 差异分布（GPU，种子 0–19，m=10，tau=inf；单位：14 维动作向量 L2，关节为弧度）",
              "分位数 50 / 75 / 90 / 95 / 99%：",
              f"- 无扰动：{_fmt(a['none'])}", f"- 推 0.06：{_fmt(a['push0_06'])}",
              f"- 推 0.06，推之前的比较：{_fmt(ap['checks_before_push'])}",
              f"- 推 0.06，推之后第一次比较：{_fmt(ap['first_check_after_push'])}",
              f"- 配对（同种子同一步，n={ap['n_paired']}）：推 > 无扰动 {ap['paired_push_gt_none']}/{ap['n_paired']}，"
              f"中位数 推 {ap['paired_median_push']} vs 无扰动 {ap['paired_median_none']}，比值中位数 {ap['paired_median_ratio']}",
              f"- 推后第一次比较超过无扰动 95% / 99% 分位数：{ap['first_after_above_none_p95']} / {ap['first_after_above_none_p99']}"
              f"（共 {len(ap['paired']) if ap['paired'] else 0} 配对；推的状态 {ap['push_status_counts']}）",
              f"- **tau 候选值**（无扰动 50/75/90/95/99%，4 位小数）：{a['tau_candidates']}"]
    (OUT / "checks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["analyze"])
    ap.parse_args(argv)
    analyze()


if __name__ == "__main__":
    main()
