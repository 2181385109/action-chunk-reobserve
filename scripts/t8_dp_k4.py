"""T8c：DP k=4 检查（结果写 results/t8_dp_k4/；一键 scripts/t8_dp_k4.sh）。

背景：T8b 的 DP 无扰动 k=4 只成功 14/50，k=8 成功 33/50，远低于官方 n_action_steps=8 的水平。是评估脚本的 bug，还是策略本身行为？

  python scripts/t8_dp_k4.py lerobot --k 4|8   # lerobot 0.3.3 自带评估（eval_main），n_action_steps=k，种子 0–49，B=50
  python scripts/t8_dp_k4.py analyze           # K1 / K2 -> results/t8_dp_k4/k_checks.json、k_checks.md、fig_k2.png

K1 逐回合比较（成功与否、步数）：我们的脚本（ours_k4，--log-traj）vs lerobot 自带评估（lerobot_k4）；
   并比较两边实际发给环境的动作（我们的 traj['action'] vs lerobot rollout 返回的 action）是否逐位相同。
   对照：k=8 同样比一次（ours = results/t8_dp_baseline/run_k8，lerobot_k8）——k=8 能对上而 k=4 对不上才说明和 k 有关。
K2 k=4 vs k=8 的动作轨迹统计（我们的 --log-traj，种子 0–49）：
   path_length / jerk / reversal_rate（metrics.py 的定义，episodes.csv 里已有）；
   “智能体在 T 块附近来回切换方向的次数”（定义见 near_t_switches 的注释）；
   另按“块边界（这一步重新预测）/ 块内部”拆分方向切换，看切换是不是发生在重新预测的那一步。
结论只能三选一：评估脚本 bug（已定位修复）/ 策略的真实行为（附证据）/ 无法定论。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunkre.config import MODELS  # noqa: E402
from chunkre.metrics import read_episodes, wilson_ci  # noqa: E402

OUT = ROOT / "results" / "t8_dp_k4"
BASE = ROOT / "results" / "t8_dp_baseline"
B, N_EP = 50, 50
OURS = {4: OUT / "ours_k4", 8: BASE / "run_k8"}
LEROBOT = {4: OUT / "lerobot_k4", 8: OUT / "lerobot_k8"}
AGENT_R = 15.0           # 智能体圆半径（PushT 坐标，baseline.json 里读出的值）
NEAR_GAP = 30.0          # “在 T 块附近”：智能体圆与 T 形块的间隙 <= 30 单位（= 2 个智能体半径）
MOVE_MIN = 1.0           # 一步移动 < 1 单位的步不算“在动”，不参与方向判定（避免静止时方向噪声）
FLIP_COS = 0.0           # 相邻两个“在动”的步，前进方向夹角 > 90°（cos < 0）算一次方向切换


# ----------------------------------------------------------------------------------------------
# lerobot 自带评估
# ----------------------------------------------------------------------------------------------
def run_lerobot(k: int) -> None:
    """在本进程里运行 lerobot 0.3.3 的 eval_main（命令行参数同 lerobot-eval），只做两处运行时替换（不改源码）：
    1) set_seed 前把 torch 后端开关设成与我们脚本相同（eval_main 自己设了 cudnn.benchmark=True）；
    2) rollout 包一层，记录 done 张量算步数、保存实际发给环境的动作。
    策略由 lerobot 自己从 hub 加载（--policy.path + --policy.n_action_steps），不经过我们的 build_policy。
    """
    import lerobot.scripts.eval as le
    import torch

    from chunkre.runner import set_determinism

    out = LEROBOT[k]
    out.mkdir(parents=True, exist_ok=True)
    captured, acts, info = [], [], {}
    orig_rollout, orig_seed, orig_eval_policy = le.rollout, le.set_seed, le.eval_policy

    def rollout(*a, **kw):
        r = orig_rollout(*a, **kw)
        captured.append(r["done"].clone())
        acts.append(r["action"].clone())
        return r

    def set_seed(s):
        info["torch_flags"] = set_determinism(True)
        orig_seed(s)

    def eval_policy(*a, **kw):
        kw["max_episodes_rendered"] = 0
        info["eval"] = orig_eval_policy(*a, **kw)
        return info["eval"]

    le.rollout, le.set_seed, le.eval_policy = rollout, set_seed, eval_policy
    lout = out / "lerobot_output"
    lout.mkdir(parents=True, exist_ok=True)
    argv = ["lerobot-eval", f"--policy.path={MODELS['dp']['repo']}", f"--policy.n_action_steps={k}",
            "--policy.device=cuda", "--env.type=pusht", f"--eval.batch_size={B}", f"--eval.n_episodes={N_EP}",
            "--seed=0", f"--output_dir={lout}"]
    sys.argv = argv
    try:
        le.init_logging()
        le.eval_main()
    finally:
        le.rollout, le.set_seed, le.eval_policy = orig_rollout, orig_seed, orig_eval_policy
    ev = info["eval"]
    np.savez_compressed(out / "actions.npz", action=torch.cat(acts, dim=0).numpy(),
                        done=torch.cat(captured, dim=0).numpy())
    done = torch.cat(captured, dim=0)
    steps = (torch.argmax(done.to(int), dim=1) + 1).tolist()
    with open(out / "episodes_lerobot.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode_ix", "seed", "success", "steps", "sum_reward", "max_reward"])
        for ep, st in zip(ev["per_episode"], steps):
            w.writerow([ep["episode_ix"], ep["seed"], int(ep["success"]), st, ep["sum_reward"], ep["max_reward"]])
    (out / "config.json").write_text(json.dumps(dict(
        what="lerobot 0.3.3 eval_main（lerobot-eval 同一入口），见 scripts/t8_dp_k4.py run_lerobot 的运行时替换",
        argv=argv, torch_flags=info.get("torch_flags"),
        actions_npz="actions.npz：lerobot rollout 返回的 action (B, T, A)，回合结束后的步被 lerobot 用 0 填充"),
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[lerobot k={k}] pc_success={ev['aggregated']['pc_success']}", flush=True)


# ----------------------------------------------------------------------------------------------
# K1
# ----------------------------------------------------------------------------------------------
def _traj(run: Path) -> list:
    return [dict(np.load(p)) for p in sorted((run / "traj").glob("batch_*.npz"))]


def k1(k: int) -> dict:
    ours = {int(r["seed"]): r for r in read_episodes(OURS[k] / "episodes.csv")}
    with open(LEROBOT[k] / "episodes_lerobot.csv", encoding="utf-8") as f:
        le = {int(r["seed"]): r for r in csv.DictReader(f)}
    assert sorted(ours) == sorted(le), "种子集合不同"
    per = []
    for s in sorted(ours):
        o, l_ = ours[s], le[s]
        per.append(dict(seed=s, ours=(int(o["success"]), int(o["steps"])), lerobot=(int(l_["success"]), int(l_["steps"])),
                        ours_max_reward=float(o["max_reward"]), lerobot_max_reward=float(l_["max_reward"])))
    n = len(per)
    # 动作逐位比较（我们的 traj['action'][:steps, i] vs lerobot rollout action[i, :steps]）
    la = np.load(LEROBOT[k] / "actions.npz")["action"]          # (B, T, A)
    seed_to_row = {s: i for i, s in enumerate(sorted(ours))}    # lerobot 的第 i 个回合 = 种子 i（start_seed=0）
    acts_same = acts_close = 0
    max_act_diff = 0.0
    first_div = []
    for z in _traj(OURS[k]):
        for i, s in enumerate(z["seeds"]):
            st = int(z["steps"][i])
            a, b = z["action"][:st, i], la[seed_to_row[int(s)], :st]
            same = np.array_equal(a, b)
            acts_same += int(same)
            d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
            max_act_diff = max(max_act_diff, d)
            acts_close += int(d < 1e-3)
            if not same:
                bad = np.flatnonzero(np.abs(a - b).max(axis=1) > 0)
                first_div.append((int(s), int(bad[0])))
    ns_o = sum(p["ours"][0] for p in per)
    ns_l = sum(p["lerobot"][0] for p in per)
    return dict(k=k, n=n, ours_success=ns_o, lerobot_success=ns_l,
                success_agree=sum(p["ours"][0] == p["lerobot"][0] for p in per),
                steps_agree=sum(p["ours"][1] == p["lerobot"][1] for p in per),
                both_agree=sum(p["ours"] == p["lerobot"] for p in per),
                actions_bitwise_identical=acts_same, actions_within_1e3=acts_close, max_abs_action_diff=max_act_diff,
                first_divergence_seed_step=first_div[:10], per_seed=per)


# ----------------------------------------------------------------------------------------------
# K2
# ----------------------------------------------------------------------------------------------
def near_t_switches(agent_pos, block_pose, repredict, steps, lp, ws_unused=None) -> dict:
    """智能体在 T 块附近来回切换方向的次数（K2 的定义，一个回合）。

    - 第 t 步智能体位移 v_t = agent_pos[t+1] − agent_pos[t]（实际位置，t = 0..steps−2）；|v_t| < MOVE_MIN 的步不算“在动”。
    - “在 T 块附近”：第 t 步智能体圆（半径 AGENT_R）与 T 形块（block_pose[t] 的位姿）的间隙 <= NEAR_GAP。
    - 方向切换：相邻两个“在动”的步 t1 < t2（中间都是不动的步也算相邻），且两步都在 T 块附近，
      cos(v_t1, v_t2) < FLIP_COS（夹角 > 90°）。
    - 同时按 t2 是不是“重新预测的那一步”（repredict[t2] 为真，即动作块的第 1 个动作）拆成块边界 / 块内部两类。
    返回 dict：n_flips、n_flips_boundary、n_flips_interior、n_near_steps（附近且在动的步数）、n_moving_steps。
    """
    from chunkre.perturb import polys_at, tee_agent_gap

    n = int(steps)
    pos = agent_pos[:n].astype(np.float64)
    blk = block_pose[:n].astype(np.float64)
    v = pos[1:] - pos[:-1]                       # (n-1, 2)
    spd = np.linalg.norm(v, axis=1)
    moving = spd >= MOVE_MIN
    near = np.array([tee_agent_gap(polys_at(lp, blk[t, 2], blk[t, :2]), pos[t], AGENT_R) <= NEAR_GAP
                     for t in range(n - 1)])
    flips = fb = fi = 0
    prev = None
    for t in range(n - 1):
        if not moving[t]:
            continue
        if prev is not None and near[prev] and near[t]:
            c = float(np.dot(v[prev], v[t]) / (spd[prev] * spd[t]))
            if c < FLIP_COS:
                flips += 1
                if bool(repredict[t]):
                    fb += 1
                else:
                    fi += 1
        prev = t
    return dict(n_flips=flips, n_flips_boundary=fb, n_flips_interior=fi,
                n_near_moving_steps=int((near & moving).sum()), n_moving_steps=int(moving.sum()), n_steps=n)


def k2_run(k: int) -> dict:
    run = OURS[k]
    geo = json.loads((BASE / "baseline.json").read_text(encoding="utf-8"))["geometry"]
    lp = [np.array(p) for p in geo["tee_rects_local"]]
    rows = {int(r["seed"]): r for r in read_episodes(run / "episodes.csv")}
    per = []
    for z in _traj(run):
        for i, s in enumerate(z["seeds"]):
            r = rows[int(s)]
            sw = near_t_switches(z["agent_pos"][:, i], z["block_pose"][:, i], z["repredict"], z["steps"][i], lp)
            per.append(dict(seed=int(s), success=int(r["success"]), path_length=float(r["path_length"]),
                            jerk=float(r["jerk"]), reversal_rate=float(r["reversal_rate"]), **sw))
    per.sort(key=lambda d: d["seed"])

    def m(key, sel=None):
        v = [d[key] for d in per if sel is None or sel(d)]
        return float(np.mean(v)) if v else float("nan")

    ns = sum(d["success"] for d in per)
    tot_near = sum(d["n_near_moving_steps"] for d in per)
    return dict(k=k, run=str(run.relative_to(ROOT)), n=len(per), n_success=ns,
                ci=list(wilson_ci(ns, len(per))),
                mean_path_length=m("path_length"), mean_jerk=m("jerk"), mean_reversal_rate=m("reversal_rate"),
                mean_flips=m("n_flips"), mean_flips_boundary=m("n_flips_boundary"), mean_flips_interior=m("n_flips_interior"),
                mean_steps=m("n_steps"), mean_near_moving_steps=m("n_near_moving_steps"),
                flips_per_100_near_steps=(100.0 * sum(d["n_flips"] for d in per) / tot_near if tot_near else float("nan")),
                mean_flips_success=m("n_flips", lambda d: d["success"] == 1),
                mean_flips_fail=m("n_flips", lambda d: d["success"] == 0),
                per_episode=per)


def analyze() -> None:
    res = dict(k1={}, k2={})
    for k in (4, 8):
        if (LEROBOT[k] / "episodes_lerobot.csv").exists() and OURS[k].exists():
            res["k1"][f"k{k}"] = k1(k)
    for k in (4, 8):
        if (OURS[k] / "traj").exists():
            res["k2"][f"k{k}"] = k2_run(k)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "k_checks.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    md = md_report(res)
    (OUT / "k_checks.md").write_text(md, encoding="utf-8", newline="\n")
    if "k4" in res["k2"] and "k8" in res["k2"]:
        fig_k2(res)
    print(md)


def md_report(res: dict) -> str:
    L = ["# T8c DP k=4 检查（种子 0–49，B=50，无扰动）", "", "## K1 我们的脚本 vs lerobot 0.3.3 自带评估（逐回合）", "",
         "| k | 成功 ours / lerobot | 成功与否一致 | 步数一致 | 成功+步数都一致 | 动作逐位相同 | 动作最大差 |", "|---|---|---|---|---|---|---|"]
    for key, x in res["k1"].items():
        L.append(f"| {x['k']} | {x['ours_success']}/{x['n']} / {x['lerobot_success']}/{x['n']} | {x['success_agree']}/{x['n']} | "
                 f"{x['steps_agree']}/{x['n']} | {x['both_agree']}/{x['n']} | {x['actions_bitwise_identical']}/{x['n']} | "
                 f"{x['max_abs_action_diff']:.3g} |")
    for key, x in res["k1"].items():
        if x["first_divergence_seed_step"]:
            L.append(f"- k={x['k']} 动作不逐位相同的回合（种子, 第一个不同的步）前 10 个：{x['first_divergence_seed_step']}")
    L += ["", "## K2 k=4 vs k=8 的动作轨迹（我们的 --log-traj）", "",
          f"定义见 `scripts/t8_dp_k4.py::near_t_switches`：T 块附近 = 智能体圆与 T 形块间隙 <= {NEAR_GAP:g}；一步位移 < {MOVE_MIN:g} 不算在动；"
          "相邻两个在动的步方向夹角 > 90° 记一次方向切换；块边界 = 切换发生在重新预测的那一步。", "",
          "| k | 成功 | path_length | jerk | reversal_rate | 方向切换次数/回合 | 其中块边界 | 块内部 | 每 100 个“附近且在动”步的切换 | 附近且在动步数/回合 | 回合步数 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, x in res["k2"].items():
        L.append(f"| {x['k']} | {x['n_success']}/{x['n']} | {x['mean_path_length']:.1f} | {x['mean_jerk']:.2f} | "
                 f"{x['mean_reversal_rate']:.3f} | {x['mean_flips']:.2f} | {x['mean_flips_boundary']:.2f} | "
                 f"{x['mean_flips_interior']:.2f} | {x['flips_per_100_near_steps']:.2f} | {x['mean_near_moving_steps']:.1f} | "
                 f"{x['mean_steps']:.1f} |")
    for key, x in res["k2"].items():
        L.append(f"- k={x['k']}：成功回合平均切换 {x['mean_flips_success']:.2f}，失败回合 {x['mean_flips_fail']:.2f}")
    L.append("")
    return "\n".join(L)


def fig_k2(res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for a, key, lab in zip(ax, ("n_flips", "jerk", "path_length"), ("direction flips near T / episode", "jerk", "path length")):
        data = [[d[key] for d in res["k2"][f"k{k}"]["per_episode"]] for k in (4, 8)]
        a.boxplot(data, tick_labels=["k=4", "k=8"], widths=0.5)
        a.set_title(lab, fontsize=9)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_k2.png", dpi=140)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lerobot")
    p.add_argument("--k", type=int, required=True, choices=[4, 8])
    sub.add_parser("analyze")
    a = ap.parse_args(argv)
    if a.cmd == "lerobot":
        run_lerobot(a.k)
    else:
        analyze()


if __name__ == "__main__":
    main()
