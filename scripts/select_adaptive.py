"""T11a 自适应规则阈值选择（事先定好的规则，机械执行；规则本身在 src/chunkre/tune_adaptive.py::select）。

  python scripts/select_adaptive.py [--dir results/tune_adaptive] [--n 50]

读 22 格的 episodes.csv（逐回合重算成功数和平均策略调用次数，不读 summary.json），按规则选出一个 (m, tau) 组合，
写 <dir>/selection.json 和 <dir>/selection.md；没有满足条件的组合时输出“无可用组合”。
格子不全（或回合数不是 --n）直接报错退出（exit 2）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunkre import tune_adaptive as ta  # noqa: E402
from chunkre.metrics import summarize_csv  # noqa: E402
from chunkre.pilot import cell_done  # noqa: E402


def load_stats(d: Path, n: int) -> dict:
    stats, missing = {}, []
    for c in ta.cells():
        if not cell_done(d / c.name, n):
            missing.append(c.name)
            continue
        m = summarize_csv(d / c.name / "episodes.csv")
        stats[c.name] = dict(n=m["n_episodes"], n_success=m["n_success"], mean_policy_calls=m["mean_policy_calls"])
    if missing:
        print(f"[select] 格子不全（或回合数不是 {n}）：{missing}", file=sys.stderr)
        sys.exit(2)
    return stats


def report(res: dict, d: Path) -> str:
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    lines = ["# T11a 自适应规则阈值选择（scripts/select_adaptive.py 生成）", "",
             f"- 参照 k=100：无扰动 {res['ref_none_success']}/{res['ref_none_n']} = {pct(res['ref_none_rate'])}，"
             f"推 0.06 {res['ref_push_success']}/{res['ref_push_n']} = {pct(res['ref_push_rate'])}，"
             f"平均策略调用 {res['ref_mean_policy_calls']:.1f}",
             f"- 可用门槛：无扰动成功率 >= {pct(res['ref_none_rate'])} − 3 个百分点 = {pct(res['none_bar'])}；"
             f"可用组合 {res['n_eligible']}/{len(res['rows'])}",
             f"- **选择结果：{res['verdict']}**" + ("（仍并列，按网格顺序取第一个；并列的还有 "
                                                  + ", ".join(res["tied_with"]) + "）" if res["still_tied"] else ""),
             "", "| 组合 | m | tau | 无扰动 | 推 0.06 | 平均策略调用（两条件） | 可用 |", "|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        mark = " ←" if r["combo"] == res["selected"] else ""
        lines.append(f"| {r['combo']}{mark} | {r['m']} | {r['tau']:g} ({r['tau_label']}) | "
                     f"{r['none_success']}/{r['none_n']} = {pct(r['none_rate'])} | "
                     f"{r['push_success']}/{r['push_n']} = {pct(r['push_rate'])} | {r['mean_policy_calls']:.1f} | "
                     f"{'是' if r['eligible'] else '否'} |")
    lines += ["", "规则：可用 = 无扰动成功率 >= k=100 无扰动 − 3pp（分数精确比较）；可用里推 0.06 成功率最高；"
              "并列选平均策略调用少的（两格平均）；仍并列取网格顺序靠前的；没有可用组合 = “无可用组合”。"
              f"数字由各格 episodes.csv 重算（目录 {d}）。调参种子上的试跑，不是正式结论。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/tune_adaptive")
    ap.add_argument("--n", type=int, default=ta.TUNE_N)
    a = ap.parse_args(argv)
    d = Path(a.dir)
    d = d if d.is_absolute() else ROOT / d
    res = ta.select(load_stats(d, a.n))
    res["n_per_cell"] = a.n
    (d / "selection.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    md = report(res, d)
    (d / "selection.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"[select] 结果：{res['verdict']}")


if __name__ == "__main__":
    main()
