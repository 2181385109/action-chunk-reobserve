"""复现结果与官方数字比较（ACT 83.0%、DP 65.4%）。

判定标准（在看到结果之前已定，原样实现）：|差| <= 7 个百分点 → 合格；7 < |差| <= 15 → 可继续但需在 README 说明；
|差| > 15 → 需排查。
另外：若 docs/official_model_cards/ 里有官方 eval_info.json（含逐回合 seed/success），对两边都有的种子计算逐种子
成功/失败一致率（只作参考，不作判定；渲染器/硬件不同，单个种子的成败本来就可能翻转）。

用法：
  python scripts/compare_official.py                         # 默认读 results/repro/act、results/repro/dp
  python scripts/compare_official.py --act DIR --dp DIR --out FILE.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunkre.metrics import read_episodes, summarize_csv  # noqa: E402

OFFICIAL = {
    "act": dict(pc=83.0, eval_info=ROOT / "docs/official_model_cards/act_aloha_sim_transfer_cube_human_eval_info.json"),
    "dp": dict(pc=65.4, eval_info=ROOT / "docs/official_model_cards/diffusion_pusht_eval_info.json"),
}
PASS_PP, WARN_PP = 7.0, 15.0


def verdict(diff_pp: float) -> str:
    a = abs(diff_pp)
    if a <= PASS_PP:
        return "合格"
    if a <= WARN_PP:
        return "可继续但需在 README 说明"
    return "需排查"


def per_seed_agreement(rows: list[dict], eval_info: Path) -> dict | None:
    if not eval_info.exists():
        return None
    off = {int(e["seed"]): bool(e["success"]) for e in json.loads(eval_info.read_text())["per_episode"]}
    ours = {int(r["seed"]): bool(int(r["success"])) for r in rows}
    common = sorted(set(off) & set(ours))
    if not common:
        return dict(n_common=0)
    agree = sum(off[s] == ours[s] for s in common)
    both = sum(off[s] and ours[s] for s in common)
    neither = sum((not off[s]) and (not ours[s]) for s in common)
    p_o = sum(off[s] for s in common) / len(common)
    p_u = sum(ours[s] for s in common) / len(common)
    return dict(
        n_common=len(common), agree=agree, agreement_rate=agree / len(common),
        both_success=both, both_fail=neither, only_official_success=sum(off[s] and not ours[s] for s in common),
        only_ours_success=sum(ours[s] and not off[s] for s in common),
        official_rate_on_common=p_o, ours_rate_on_common=p_u,
        chance_agreement_if_independent=p_o * p_u + (1 - p_o) * (1 - p_u),
        note="只作参考，不作判定",
    )


def compare(model: str, run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    m = summarize_csv(run_dir / "episodes.csv")   # 直接由 CSV 重算，不信任手写数字
    ours_pc = 100 * m["success_rate"]
    diff = ours_pc - OFFICIAL[model]["pc"]
    return dict(
        model=model, run_dir=str(run_dir), n_episodes=m["n_episodes"], ours_pc=ours_pc,
        ours_ci95_pc=[100 * x for x in m["success_ci95_wilson"]], official_pc=OFFICIAL[model]["pc"],
        diff_pp=diff, verdict=verdict(diff),
        per_seed=per_seed_agreement(read_episodes(run_dir / "episodes.csv"), OFFICIAL[model]["eval_info"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default=str(ROOT / "results/repro/act"))
    ap.add_argument("--dp", default=str(ROOT / "results/repro/dp"))
    ap.add_argument("--out", default=None, help="默认 <act 的上一级目录>/compare_official.json")
    a = ap.parse_args()
    res = {}
    for model, d in [("act", a.act), ("dp", a.dp)]:
        if (Path(d) / "episodes.csv").exists():
            res[model] = compare(model, Path(d))
        else:
            res[model] = dict(error=f"{d}/episodes.csv 不存在")
    res["criteria"] = f"|差|<={PASS_PP:g}pp 合格；{PASS_PP:g}<|差|<={WARN_PP:g} 可继续但需在 README 说明；>{WARN_PP:g} 需排查"
    out = Path(a.out) if a.out else Path(a.act).parent / "compare_official.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for model in ["act", "dp"]:
        r = res[model]
        if "error" in r:
            print(f"[{model}] {r['error']}")
            continue
        ps = r["per_seed"]
        extra = ""
        if ps and ps.get("n_common"):
            extra = (f"；逐种子一致率 {ps['agreement_rate']:.1%}（{ps['agree']}/{ps['n_common']}，"
                     f"若独立则约 {ps['chance_agreement_if_independent']:.1%}，仅参考）")
        print(f"[{model}] 我们 {r['ours_pc']:.1f}%（{r['n_episodes']} 回合，95% CI {r['ours_ci95_pc'][0]:.1f}–"
              f"{r['ours_ci95_pc'][1]:.1f}）vs 官方 {r['official_pc']}%：差 {r['diff_pp']:+.1f} 个百分点 → "
              f"{r['verdict']}{extra}")
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
