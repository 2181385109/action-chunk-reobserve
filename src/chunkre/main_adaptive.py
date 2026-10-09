"""T11b 自适应规则正式评估（PREREGISTRATION_ADAPTIVE.md 的实现）：2 格 ad__none、ad__push0.06。纯计算，不依赖仿真，单元测试覆盖。

参数写死为登记的 m=10、tau=0.2906、h_cmp=20、min_gap=20（登记 §2）；run_main_adaptive.sh 开跑前把“要跑的配置”与登记文件里的
数字逐项比较，不一致就一个回合都不跑（`registered_params()` 直接从 PREREGISTRATION_ADAPTIVE.md 的文字里解析，不是抄一份常量）。
对照 = results/main 里的 k100__none、k100__push0.06（同一批种子 200000–200199，不重跑）。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .pilot import PILOT_PUSH_PARAMS

ROOT = Path(__file__).resolve().parents[2]
PREREG_NAME = "PREREGISTRATION_ADAPTIVE.md"

# 登记的参数（登记 §2）。单元测试核对它与 PREREGISTRATION_ADAPTIVE.md 里解析出的数字、tune_adaptive 配置里 q90 的 tau 一致
ADAPTIVE_PARAMS = dict(m=10, tau=0.2906, h_cmp=20, min_gap=20)
PARAM_KEYS = ("m", "tau", "h_cmp", "min_gap")

ADAPTIVE_SEEDS = "200000-200199"   # 与 results/main 同一批种子；每格 200 回合（按种子配对）
ADAPTIVE_N = 200
ADAPTIVE_BATCH = 10
CONTROL_DIR = "results/main"       # 对照所在目录（格子名 k100__none 等）
CONTROL_POLICY = "k100"
FIXED_POLICIES = ["k20", "k50", "k100", "te"]    # 描述性对比（登记 §6.1）
NONINF_MARGIN = 0.05               # HA2 非劣边际（登记 §5）：下界 > −0.05

# (标签, perturb.kind, 强度)；格子名 "ad__<标签>"，对照格名 "k100__<标签>"
CONDITIONS = [("none", "none", 0.0), ("push0.06", "push", 0.06)]
POLICY = "ad"


def cell_names() -> list[str]:
    return [f"{POLICY}__{tag}" for tag, _, _ in CONDITIONS]


def control_names() -> list[str]:
    return [f"{CONTROL_POLICY}__{tag}" for tag, _, _ in CONDITIONS]


def config_yaml(tag: str, params: dict | None = None, seeds: str = ADAPTIVE_SEEDS, batch_size: int = ADAPTIVE_BATCH,
                max_steps: int | None = None) -> str:
    p = dict(ADAPTIVE_PARAMS if params is None else params)
    kind, strength = {t: (k, s) for t, k, s in CONDITIONS}[tag]
    lines = [f"# T11b 自适应规则正式评估格子 ad__{tag}（由 scripts/main_adaptive_tools.py gen 生成，不要手改）",
             f"name: {POLICY}__{tag}", "model: act", f"seeds: {seeds}", f"batch_size: {batch_size}"]
    if max_steps:
        lines.append(f"max_steps: {int(max_steps)}   # 只用于脚本自检")
    lines += ["mode: adaptive", "adaptive:", f"  m: {p['m']}", f"  tau: {p['tau']!r}", f"  h_cmp: {p['h_cmp']}",
              f"  min_gap: {p['min_gap']}"]
    if kind != "none":
        lines += ["perturb:", f"  kind: {kind}", f"  strength: {strength:g}", "  params:"]
        lines += [f"    {k}: {v}" for k, v in PILOT_PUSH_PARAMS.items()]
    return "\n".join(lines) + "\n"


def write_configs(out_dir: Path, params: dict | None = None, seeds: str = ADAPTIVE_SEEDS,
                  batch_size: int = ADAPTIVE_BATCH) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for tag, _, _ in CONDITIONS:
        p = out_dir / f"{POLICY}__{tag}.yaml"
        p.write_text(config_yaml(tag, params, seeds, batch_size), encoding="utf-8", newline="\n")
        paths.append(p)
    return paths


# ---------------------------------------------------------------- 登记里的参数
_RE_MT = re.compile(r"\*\*m = (\d+); tau = ([0-9.]+)\*\*")
_RE_HG = re.compile(r"h_cmp = (\d+); min_gap = (\d+); L = (\d+)")


def registered_params(root: Path = ROOT) -> dict:
    """从 PREREGISTRATION_ADAPTIVE.md §2 “Fixed parameters” 那一段的文字里解析 m、tau、h_cmp、min_gap（数值）。找不到就抛错。"""
    text = (Path(root) / PREREG_NAME).read_text(encoding="utf-8")
    i = text.find("**Fixed parameters:**")
    if i < 0:
        raise ValueError(f"{PREREG_NAME} 里找不到 “Fixed parameters” 段")
    seg = text[i:i + 700]
    a, b = _RE_MT.search(seg), _RE_HG.search(seg)
    if not (a and b):
        raise ValueError(f"{PREREG_NAME} 的 “Fixed parameters” 段解析不出 m / tau / h_cmp / min_gap")
    return dict(m=int(a.group(1)), tau=float(a.group(2)), h_cmp=int(b.group(1)), min_gap=int(b.group(2)))


def _parse_yaml_adaptive(text: str) -> dict:
    """读生成的 YAML 里 adaptive: 下的 m / tau / h_cmp / min_gap（只认 write_configs 写出的格式；不依赖 yaml 库的行为差异）。"""
    import yaml

    d = yaml.safe_load(text) or {}
    ad = d.get("adaptive")
    if d.get("mode") != "adaptive" or not isinstance(ad, dict):
        raise ValueError("配置不是 mode: adaptive")
    return {k: ad.get(k) for k in PARAM_KEYS}


def check_run_params(config_paths, reg: dict) -> list[str]:
    """要跑的每个配置（YAML 文件）的 adaptive 参数与登记逐项比较。返回不一致的描述（空 = 全部一致）。"""
    bad = []
    for p in config_paths:
        try:
            got = _parse_yaml_adaptive(Path(p).read_text(encoding="utf-8"))
        except Exception as e:   # noqa: BLE001
            bad.append(f"{Path(p).name}: 无法读取自适应参数（{e}）")
            continue
        for k in PARAM_KEYS:
            g, r = got.get(k), reg[k]
            same = (g is not None) and (float(g) == float(r))
            if not same:
                bad.append(f"{Path(p).name}: {k} = {g!r}，登记为 {r!r}")
    return bad


# ---------------------------------------------------------------- 读自适应记录
def load_replans(run_dir: Path) -> list[dict]:
    """读 <run_dir>/adaptive/batch_*.npz 的换计划记录：逐条 dict(seed, t, cause)（cause：0 初始 / 1 块用完 / 2 差异触发）。"""
    out = []
    for p in sorted((Path(run_dir) / "adaptive").glob("batch_*.npz")):
        z = np.load(p)
        seeds = z["seeds"]
        for e, t, c in zip(z["rp_env"], z["rp_t"], z["rp_cause"]):
            out.append(dict(seed=int(seeds[e]), t=int(t), cause=int(c)))
    return out


def first_reaction_steps(rows: list[dict], replans: list[dict]) -> dict:
    """推 0.06 格：被推之后第一次“差异触发”的换计划距推的步数。

    推发生在 push_step 步“策略选好动作之后、env.step 之前”，策略最早在 push_step+1 步才看到被推后的画面，
    所以只数 t > push_step 的差异触发换计划（t == push_step 的那次用的是推之前的画面，不算反应；个数单独报告）。
    只看 push_status == "pushed" 的回合。返回 steps（每个有反应的回合一个数）、n_pushed、n_no_reaction、n_at_push_step。
    """
    by_seed: dict[int, list[int]] = {}
    for r in replans:
        if r["cause"] == 2:
            by_seed.setdefault(r["seed"], []).append(r["t"])
    steps, n_pushed, n_none, n_at = [], 0, 0, 0
    for row in rows:
        if row.get("push_status") != "pushed":
            continue
        n_pushed += 1
        ps = int(float(row["push_step"]))
        ts = sorted(by_seed.get(int(float(row["seed"])), []))
        after = [t for t in ts if t > ps]
        n_at += int(ps in ts)
        if after:
            steps.append(after[0] - ps)
        else:
            n_none += 1
    return dict(steps=steps, n_pushed=n_pushed, n_no_reaction=n_none, n_replan_at_push_step=n_at)


__all__ = ["PREREG_NAME", "ADAPTIVE_PARAMS", "PARAM_KEYS", "ADAPTIVE_SEEDS", "ADAPTIVE_N", "ADAPTIVE_BATCH", "CONTROL_DIR",
           "CONTROL_POLICY", "FIXED_POLICIES", "NONINF_MARGIN", "CONDITIONS", "POLICY", "cell_names", "control_names",
           "config_yaml", "write_configs", "registered_params", "check_run_params", "load_replans", "first_reaction_steps"]
