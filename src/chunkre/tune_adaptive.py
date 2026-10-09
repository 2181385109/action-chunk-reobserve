"""T11a 自适应规则阈值试跑网格 + 选择规则（事先定）。纯计算，不依赖仿真，单元测试覆盖。

网格：m ∈ {10, 20} × tau ∈ 5 个候选值 × 条件 {无扰动, 推 0.06 m} = 20 格，加 k=100 在同样两种条件下的 2 格参照，共 22 格。
种子 6000–6049（调参区间里此前没用过的新种子），B=10，GPU 渲染。h_cmp = 20、min_gap = 20 固定。
tau 候选值 = T11a A3 里“无扰动”（GPU，种子 0–19，m=10，tau=inf）全部比较差异值的 50/75/90/95/99% 分位数，
四舍五入到 4 位小数（来源 results/t11_checks/checks.json 的 a3.tau_candidates；单元测试核对两者一致）。

选择规则（scripts/select_adaptive.py 机械执行）：
  1. 可用 = 无扰动成功率 >= k=100 无扰动成功率 − 3 个百分点（用分数精确比较，不用浮点）；
  2. 在可用组合里选推 0.06 成功率最高的；
  3. 并列时选平均策略调用次数少的（两种条件全部回合的平均 = 两格平均调用次数的平均，两格回合数相同）；
  4. 仍并列：取网格顺序靠前的（m 小、tau 小），并在输出里标“仍并列”；
  5. 没有可用组合：输出“无可用组合”。
格子名："ad_m<m>_<q>__<条件>"（如 ad_m10_q90__push0.06），参照 "k100__none" / "k100__push0.06"。
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .pilot import PILOT_PUSH_PARAMS

TUNE_SEEDS = "6000-6049"
TUNE_N = 50
TUNE_BATCH = 10
M_VALUES = (10, 20)
H_CMP = 20
MIN_GAP = 20
TAU_QUANTILES = ("q50", "q75", "q90", "q95", "q99")
# A3 无扰动差异分布的 50/75/90/95/99% 分位数（弧度级；results/t11_checks/checks.json a3.tau_candidates）
TAU_CANDIDATES = (0.0919, 0.1784, 0.2906, 0.3964, 0.7032)
CONDITIONS = [("none", "none", 0.0), ("push0.06", "push", 0.06)]   # (标签, perturb.kind, 强度)
REF_POLICY = "k100"
NONE_MARGIN = Fraction(3, 100)      # 3 个百分点


@dataclass(frozen=True)
class TuneCell:
    name: str
    policy: str             # "k100" 或 "ad_m10_q90"
    cond: str               # none / push0.06
    kind: str               # perturb.kind
    strength: float
    m: int | None = None    # 参照格为 None
    tau_label: str | None = None
    tau: float | None = None

    @property
    def adaptive(self) -> bool:
        return self.m is not None


def taus() -> list[tuple[str, float]]:
    if any(t is None for t in TAU_CANDIDATES):
        raise RuntimeError("TAU_CANDIDATES 还没填（需要先跑 T11a A3：scripts/t11_checks.sh）")
    return list(zip(TAU_QUANTILES, TAU_CANDIDATES))


def combos() -> list[tuple[int, str, float]]:
    """20 格对应的 10 个 (m, tau 标签, tau) 组合，网格顺序（m 小在前，tau 小在前）。"""
    return [(m, q, t) for m in M_VALUES for q, t in taus()]


def combo_name(m: int, q: str) -> str:
    return f"ad_m{m}_{q}"


def cells() -> list[TuneCell]:
    """22 格：先两格 k=100 参照，再每个组合的无扰动 / 推 0.06。"""
    out = [TuneCell(f"{REF_POLICY}__{c}", REF_POLICY, c, kind, s) for c, kind, s in CONDITIONS]
    for m, q, t in combos():
        for c, kind, s in CONDITIONS:
            out.append(TuneCell(f"{combo_name(m, q)}__{c}", combo_name(m, q), c, kind, s, m, q, t))
    return out


def config_yaml(c: TuneCell, seeds: str = TUNE_SEEDS, batch_size: int = TUNE_BATCH,
                max_steps: int | None = None) -> str:
    lines = [f"# T11a 自适应阈值试跑格子 {c.name}（由 scripts/tune_adaptive_tools.py gen 生成，不要手改）",
             f"name: {c.name}", "model: act", f"seeds: {seeds}", f"batch_size: {batch_size}"]
    if max_steps:
        lines.append(f"max_steps: {int(max_steps)}   # 只用于试跑脚本自检")
    if c.adaptive:
        lines += ["mode: adaptive", "adaptive:", f"  m: {c.m}", f"  tau: {c.tau!r}   # {c.tau_label}",
                  f"  h_cmp: {H_CMP}", f"  min_gap: {MIN_GAP}"]
    else:
        lines += ["mode: fixed", "k: 100"]
    if c.kind != "none":
        lines += ["perturb:", f"  kind: {c.kind}", f"  strength: {c.strength:g}", "  params:"]
        lines += [f"    {k}: {v}" for k, v in PILOT_PUSH_PARAMS.items()]
    return "\n".join(lines) + "\n"


def select(stats: dict[str, dict]) -> dict:
    """选择规则。stats：格子名 -> {n, n_success, mean_policy_calls}（22 格都要有）。返回结果 dict。"""
    missing = [c.name for c in cells() if c.name not in stats]
    if missing:
        raise ValueError(f"缺少格子：{missing}")

    def rate(name):
        s = stats[name]
        return Fraction(int(s["n_success"]), int(s["n"]))

    ref_none = rate(f"{REF_POLICY}__none")
    bar = ref_none - NONE_MARGIN
    rows = []
    for m, q, t in combos():
        cn = combo_name(m, q)
        none, push = stats[f"{cn}__none"], stats[f"{cn}__push0.06"]
        calls = (float(none["mean_policy_calls"]) + float(push["mean_policy_calls"])) / 2
        rows.append(dict(combo=cn, m=m, tau_label=q, tau=t,
                         none_success=int(none["n_success"]), none_n=int(none["n"]), none_rate=float(rate(f"{cn}__none")),
                         push_success=int(push["n_success"]), push_n=int(push["n"]), push_rate=float(rate(f"{cn}__push0.06")),
                         mean_policy_calls=calls, eligible=bool(rate(f"{cn}__none") >= bar),
                         _push=rate(f"{cn}__push0.06")))
    ok = [r for r in rows if r["eligible"]]
    out = dict(ref_none_rate=float(ref_none), ref_none_success=int(stats[f"{REF_POLICY}__none"]["n_success"]),
               ref_none_n=int(stats[f"{REF_POLICY}__none"]["n"]),
               ref_push_rate=float(rate(f"{REF_POLICY}__push0.06")),
               ref_push_success=int(stats[f"{REF_POLICY}__push0.06"]["n_success"]),
               ref_push_n=int(stats[f"{REF_POLICY}__push0.06"]["n"]),
               ref_mean_policy_calls=(float(stats[f"{REF_POLICY}__none"]["mean_policy_calls"])
                                      + float(stats[f"{REF_POLICY}__push0.06"]["mean_policy_calls"])) / 2,
               none_bar=float(bar), n_eligible=len(ok), selected=None, still_tied=False, tied_with=[])
    if ok:
        best_push = max(r["_push"] for r in ok)
        top = [r for r in ok if r["_push"] == best_push]
        min_calls = min(r["mean_policy_calls"] for r in top)
        top2 = [r for r in top if r["mean_policy_calls"] == min_calls]   # 网格顺序保持
        out["selected"] = top2[0]["combo"]
        out["still_tied"] = len(top2) > 1
        out["tied_with"] = [r["combo"] for r in top2[1:]]
        out["tied_on_push"] = [r["combo"] for r in top]
    for r in rows:
        r.pop("_push")
    out["rows"] = rows
    out["verdict"] = out["selected"] if out["selected"] else "无可用组合"
    return out


__all__ = ["TUNE_SEEDS", "TUNE_N", "TUNE_BATCH", "M_VALUES", "H_CMP", "MIN_GAP", "TAU_QUANTILES", "TAU_CANDIDATES",
           "CONDITIONS", "REF_POLICY", "NONE_MARGIN", "TuneCell", "taus", "combos", "combo_name", "cells",
           "config_yaml", "select"]
