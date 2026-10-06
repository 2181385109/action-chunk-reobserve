"""T2 正确性检查 V1–V4 的运行与判定（结果写 results/t2_checks/）。

用法（WSL 内，source scripts/env.sh 之后；一般用 scripts/t2_checks.sh 一键跑）：
  python scripts/t2_checks.py run act      # 在同一进程里跑 ACT 的 V1 第二次、V2、V4、te、V3
  python scripts/t2_checks.py run dp       # 同上，DP
  python scripts/t2_checks.py analyze      # 读所有结果，判定通过/未通过，写 checks.json / checks.md
V1 的第一次运行由 scripts/eval.py 在另一个进程里完成（<model>_v1_run1），所以 V1 同时检验“跨进程”可复现。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunkre.config import load_config  # noqa: E402
from chunkre.metrics import TIME_COLUMNS, read_episodes  # noqa: E402

OUT = ROOT / "results" / "t2_checks"
SEEDS, B = "0-9", 10
DEFAULT_K = {"act": 100, "dp": 8}
V4_KS = {"act": [1, 10, 100], "dp": [1, 8, 15]}
DP_K1_MAX_STEPS = 30
REWARD_COLS = ["sum_reward", "max_reward"]


def cfg_for(model, **kw):
    return load_config(ROOT / "configs" / f"{model}_default.yaml", dict(seeds=SEEDS, batch_size=B, **kw))


def run_lerobot_eval(model: str, out_dir: Path, allow_cpu_render: bool = False) -> None:
    """V3：用 lerobot 0.3.3 自带的 eval_policy（同样的种子 0–9、B=10、全局种子 0）跑一遍。

    eval_policy 只返回 sum/max_reward、success、seed，不返回步数。为拿到步数，把模块里的 rollout
    换成一个包装函数（运行时替换模块属性，不改 LeRobot 源码），记录每批的 done 张量：步数 = 第一个 done 的下标 + 1。
    """
    import lerobot.scripts.eval as le
    import torch

    from chunkre.runner import (build_policy, check_renderer, gl_renderer_of, make_vec_env, set_determinism,
                                set_global_seed)

    cfg = cfg_for(model)
    flags = set_determinism(cfg.deterministic)       # 与我们脚本相同的后端开关
    policy, pinfo = build_policy(cfg)
    env = make_vec_env(model, B)
    captured = []
    orig = le.rollout

    def wrapped(*a, **kw):
        r = orig(*a, **kw)
        captured.append(r["done"].clone())
        return r

    le.rollout = wrapped
    try:
        set_global_seed(cfg.seeds[0])                 # 对应 eval_main 里的 set_seed(cfg.seed)，start_seed=0
        with torch.no_grad():
            info = le.eval_policy(env, policy, n_episodes=len(cfg.seeds), start_seed=cfg.seeds[0])
        gl_renderer = gl_renderer_of(env)
        check_renderer(model, gl_renderer, allow_cpu_render)
    finally:
        le.rollout = orig
        env.close()
    done = torch.cat(captured, dim=0)
    steps = (torch.argmax(done.to(int), dim=1) + 1).tolist()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    with open(out_dir / "episodes_lerobot.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode_ix", "seed", "success", "steps", "sum_reward", "max_reward"])
        for ep, st in zip(info["per_episode"], steps):
            w.writerow([ep["episode_ix"], ep["seed"], int(ep["success"]), st, ep["sum_reward"], ep["max_reward"]])
    (out_dir / "config.json").write_text(json.dumps(dict(
        what="lerobot.scripts.eval.eval_policy，n_episodes=10，env batch=10，start_seed=0，调用前 set_seed(0)",
        policy=pinfo, torch_flags=flags, seeds=cfg.seeds, gl_renderer=gl_renderer), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[V3 {model}] lerobot pc_success={info['aggregated']['pc_success']}", flush=True)


def run_model(model: str, only: list[str] | None = None, cpu: bool = False) -> None:
    """only：要跑的检查子集（v1b, v2, v4, v3）；cpu=True 时结果目录加前缀 <model>_cpu_，并允许 CPU 渲染
    （调用前需 export CHUNK_RENDER=cpu 再 source env.sh）。"""
    from chunkre.runner import run_eval

    only = only or ["v1b", "v2", "v4", "v3"]
    pre = f"{model}_cpu_" if cpu else f"{model}_"
    jobs = []
    if "v1b" in only:
        jobs.append(("v1_run2", dict()))
    if "v2" in only:
        jobs.append(("v2_k_explicit", dict(k=DEFAULT_K[model])))
    if "v4" in only:
        for k in V4_KS[model]:
            if k == DEFAULT_K[model]:
                continue  # 默认 k 的结果就是 v1_run1
            kw = dict(k=k)
            if model == "dp" and k == 1:
                kw["max_steps"] = DP_K1_MAX_STEPS
            jobs.append((f"v4_k{k}", kw))
        if model == "act":
            jobs.append(("v4_te", dict(mode="te")))
    for name, kw in jobs:
        cfg = cfg_for(model, name=f"{pre}{name}", allow_cpu_render=cpu, **kw)
        run_eval(cfg, OUT / f"{pre}{name}", log=lambda s: print(s, flush=True))
    if "v3" in only:
        run_lerobot_eval(model, OUT / f"{pre}v3_lerobot", allow_cpu_render=cpu)


# ---------------------------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------------------------
def _rows(name):
    p = OUT / name / "episodes.csv"
    return read_episodes(p) if p.exists() else None


def compare_rows(a, b):
    """逐行比较（除时间列）。返回 (完全一致?, 不一致的列 -> 行数, 奖励列最大绝对差)。"""
    if a is None or b is None or len(a) != len(b):
        return False, {"_missing_or_len": 1}, float("nan")
    bad, max_rd = {}, 0.0
    for ra, rb in zip(a, b):
        for c in ra:
            if c in TIME_COLUMNS:
                continue
            if ra[c] != rb[c]:
                bad[c] = bad.get(c, 0) + 1
                if c in REWARD_COLS:
                    max_rd = max(max_rd, abs(float(ra[c]) - float(rb[c])))
    return not bad, bad, max_rd


def v1v2(model, a_name, b_name):
    a, b = _rows(a_name), _rows(b_name)
    exact, bad, max_rd = compare_rows(a, b)
    only_reward_tail = (not exact) and set(bad) <= set(REWARD_COLS) and max_rd <= 1e-9
    return dict(a=a_name, b=b_name, n_rows=len(a or []), exact=exact, mismatched_columns=bad,
                max_abs_reward_diff=max_rd, only_reward_float_tail=only_reward_tail,
                passed=bool(exact or only_reward_tail))


def v3(pre):
    ours = _rows(f"{pre}v1_run1")
    p = OUT / f"{pre}v3_lerobot" / "episodes_lerobot.csv"
    if ours is None or not p.exists():
        return dict(passed=False, error="缺结果")
    with open(p, newline="", encoding="utf-8") as f:
        le = {int(r["seed"]): r for r in csv.DictReader(f)}
    n_succ = n_steps = n_both = 0
    sumr_d, maxr_d, per = [], [], []
    for r in ours:
        s = int(r["seed"])
        L = le[s]
        ok_s = int(r["success"]) == int(L["success"])
        ok_t = int(r["steps"]) == int(L["steps"])
        n_succ += ok_s
        n_steps += ok_t
        n_both += ok_s and ok_t
        sumr_d.append(abs(float(r["sum_reward"]) - float(L["sum_reward"])))
        maxr_d.append(abs(float(r["max_reward"]) - float(L["max_reward"])))
        per.append(dict(seed=s, ours=(int(r["success"]), int(r["steps"])),
                        lerobot=(int(L["success"]), int(L["steps"]))))
    n = len(ours)
    return dict(n=n, success_agree=n_succ, steps_agree=n_steps, both_agree=n_both,
                ours_success_rate=sum(int(r["success"]) for r in ours) / n,
                lerobot_success_rate=sum(int(L["success"]) for L in le.values()) / len(le),
                max_abs_sum_reward_diff=max(sumr_d), max_abs_max_reward_diff=max(maxr_d),
                per_seed=per, passed=n_both == n)


def v4(model):
    k_runs = {}
    for k in V4_KS[model]:
        name = f"{model}_v1_run1" if k == DEFAULT_K[model] else f"{model}_v4_k{k}"
        k_runs[k] = (name, _rows(name))
    res = dict(per_k={}, passed=True)
    for k, (name, rows) in k_runs.items():
        if rows is None:
            res["per_k"][k] = dict(run=name, error="缺结果")
            res["passed"] = False
            continue
        exact = sum(int(r["n_policy_calls"]) == math.ceil(int(r["steps"]) / k) for r in rows)
        mc = sum(int(r["n_policy_calls"]) for r in rows) / len(rows)
        ms = sum(int(r["steps"]) for r in rows) / len(rows)
        res["per_k"][k] = dict(run=name, mean_calls=mc, mean_steps=ms, mean_steps_over_k=ms / k,
                               n_calls_eq_ceil_steps_over_k=exact, n=len(rows),
                               capped=sum(int(r["capped"]) for r in rows),
                               success=sum(int(r["success"]) for r in rows),
                               mean_smoothness=sum(float(r["smoothness"]) for r in rows) / len(rows))
        res["passed"] &= exact == len(rows)
    # 逐回合结果三者不应全同（用 成功、步数、平滑度 比较）
    sig = {k: [(r["success"], r["steps"], r["smoothness"]) for r in rows]
           for k, (_, rows) in k_runs.items() if rows}
    if len(sig) == len(V4_KS[model]):
        ks = list(sig)
        all_same = [len({sig[k][i] for k in ks}) == 1 for i in range(len(sig[ks[0]]))]
        res["n_episodes_identical_across_ks"] = sum(all_same)
        res["not_all_identical"] = not all(all_same)
        res["passed"] &= res["not_all_identical"]
    if model == "act":
        rows = _rows("act_v4_te")
        if rows is None:
            res["te"] = dict(error="缺结果")
            res["passed"] = False
        else:
            ok = sum(int(r["n_policy_calls"]) == int(r["steps"]) for r in rows)
            res["te"] = dict(n=len(rows), success=sum(int(r["success"]) for r in rows),
                             n_calls_eq_steps=ok,
                             mean_steps=sum(int(r["steps"]) for r in rows) / len(rows))
            res["passed"] &= ok == len(rows)
    return res


def analyze():
    from chunkre.metrics import summarize_csv
    from summarize import same

    # 变体：act（GPU 渲染）、act_cpu（CPU 渲染，只做 V1–V3，用来检验脚本本身）、dp
    out = {}
    for m, pre in [("act", "act_"), ("act_cpu", "act_cpu_"), ("dp", "dp_")]:
        model = m.split("_")[0]
        if not (OUT / f"{pre}v1_run1").exists():
            continue
        out[m] = dict(
            V1=v1v2(model, f"{pre}v1_run1", f"{pre}v1_run2"),
            V2=v1v2(model, f"{pre}v1_run1", f"{pre}v2_k_explicit"),
            V3=v3(pre),
        )
        if m != "act_cpu":
            out[m]["V4"] = v4(model)
    # 每个结果目录的 summary 可由 CSV 重算
    resum = {}
    for d in sorted(OUT.iterdir()):
        if (d / "episodes.csv").exists() and (d / "summary.json").exists():
            s = json.loads((d / "summary.json").read_text(encoding="utf-8"))["metrics"]
            r = summarize_csv(d / "episodes.csv")
            resum[d.name] = all(same(s.get(k2), v) for k2, v in r.items())
    out["summary_recomputable"] = resum
    (OUT / "checks.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# T2 正确性检查结果（scripts/t2_checks.py analyze 自动生成）", ""]
    titles = {"act": "ACT（GPU 渲染 D3D12）", "act_cpu": "ACT（CPU 渲染 llvmpipe，只做 V1–V3）", "dp": "DP"}
    for m in [x for x in titles if x in out]:
        o = out[m]
        lines.append(f"## {titles[m]}")
        for v in ["V1", "V2"]:
            x = o[v]
            lines.append(f"- {v} {'通过' if x['passed'] else '未通过'}：{x['a']} vs {x['b']}，{x['n_rows']} 行，"
                         f"完全一致={x['exact']}，不一致列={x['mismatched_columns']}，"
                         f"奖励最大绝对差={x['max_abs_reward_diff']:.3g}")
        x = o["V3"]
        if "error" in x:
            lines.append(f"- V3 未通过：{x['error']}")
        else:
            lines.append(f"- V3 {'通过' if x['passed'] else '未通过'}：成功一致 {x['success_agree']}/{x['n']}，"
                         f"步数一致 {x['steps_agree']}/{x['n']}，两者都一致 {x['both_agree']}/{x['n']}；"
                         f"成功率 ours {x['ours_success_rate']:.2f} / lerobot {x['lerobot_success_rate']:.2f}；"
                         f"sum_reward 最大差 {x['max_abs_sum_reward_diff']:.4g}，max_reward 最大差 "
                         f"{x['max_abs_max_reward_diff']:.4g}")
        x = o.get("V4")
        if x is None:
            lines.append("")
            continue
        lines.append(f"- V4 {'通过' if x['passed'] else '未通过'}：")
        for k, d in x["per_k"].items():
            if "error" in d:
                lines.append(f"  - k={k}: {d['error']}")
                continue
            lines.append(f"  - k={k}（{d['run']}）：平均调用 {d['mean_calls']:.2f}，平均步数/k {d['mean_steps_over_k']:.2f}，"
                         f"调用=ceil(步数/k) {d['n_calls_eq_ceil_steps_over_k']}/{d['n']}，成功 {d['success']}/{d['n']}，"
                         f"截断 {d['capped']}，平均平滑度 {d['mean_smoothness']:.4g}")
        if "not_all_identical" in x:
            lines.append(f"  - 三个 k 下逐回合（成功,步数,平滑度）完全相同的回合数：{x['n_episodes_identical_across_ks']}/10"
                         f"（不应三者全同：{'满足' if x['not_all_identical'] else '不满足'}）")
        if "te" in x and "error" not in x["te"]:
            t = x["te"]
            lines.append(f"  - te：跑通 1 批，成功 {t['success']}/{t['n']}，调用=步数 {t['n_calls_eq_steps']}/{t['n']}")
        lines.append("")
    lines.append(f"summary 可由 CSV 重算：{sum(resum.values())}/{len(resum)} 个目录一致 {resum}")
    (OUT / "checks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("model", nargs="?", choices=["act", "dp"])
    ap.add_argument("--only", default=None, help="逗号分隔：v1b,v2,v4,v3")
    ap.add_argument("--cpu", action="store_true", help="CPU 渲染变体（先 export CHUNK_RENDER=cpu）")
    a = ap.parse_args()
    if a.cmd == "run":
        run_model(a.model, a.only.split(",") if a.only else None, a.cpu)
    else:
        analyze()


if __name__ == "__main__":
    main()
