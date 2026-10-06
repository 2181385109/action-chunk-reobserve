"""T3：ACT k=1 诊断 Q1–Q4（结果写 results/t3_k1_diag/；一般用 scripts/t3_diag.sh 一键跑）。

  python scripts/t3_diag.py q1-lerobot   # Q1 的 lerobot 一侧：lerobot 0.3.3 自带 eval_main（n_action_steps=1），CPU 渲染
  python scripts/t3_diag.py q2           # Q2/Q4：GPU 渲染，种子 0–19，B=10，k ∈ {1,2,5,10,100} 和 te，打开 --log-traj
  python scripts/t3_diag.py analyze      # 读所有结果 → diag.json / diag.md
Q1 的我们一侧由 scripts/eval.py 跑两次（--log-traj 关 / 开），见 t3_diag.sh。
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

from chunkre.config import MODELS, load_config  # noqa: E402
from chunkre.metrics import TIME_COLUMNS, read_episodes  # noqa: E402

OUT = ROOT / "results" / "t3_k1_diag"
Q1_SEEDS, Q2_SEEDS, B = "0-9", "0-19", 10
Q2_CONDS = [("k1", dict(k=1)), ("k2", dict(k=2)), ("k5", dict(k=5)), ("k10", dict(k=10)),
            ("k100", dict(k=100)), ("te", dict(mode="te"))]
Q4_FRAME_CONDS = ("k1", "k100")
Q3_OFFSETS = (0, 4, 9, 19)          # 块里第 1、5、10、20 个动作
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]   # aloha 14 维里的 12 个手臂关节（6、13 是夹爪）
GRIPPERS = [6, 13]                  # 左、右夹爪（归一化开合度：1 张开、0 闭合）
LIFT_REF_STEP = 10                  # 方块初始高度 0.05 m，落到桌面（约 0.02 m）后才开始算抬升：以第 10 步高度为基准
LATE_STEP = 200                     # “后半程”：第 200 步以后
STILL_EPS = 1e-4                    # 每步手臂关节移动 < 1e-4 rad 视为静止
STALL_RADIUS = 0.01                 # 停滞：此后一直在最终位置 0.01 rad 以内
# 首次夹爪闭合：某个夹爪的观测开合度（1 张开、0 闭合；初始约 0.1）先张开到 >= GRIP_OPEN，
# 之后第一次比此前最大值回落 >= GRIP_CLOSE_DROP 的时间步（k=100 成功回合里抓住方块时约从 0.97 落到 0.6–0.68）
GRIP_OPEN = 0.8
GRIP_CLOSE_DROP = 0.2


# ----------------------------------------------------------------------------------------------
# Q1：lerobot 自带评估（eval_main），n_action_steps=1
# ----------------------------------------------------------------------------------------------
def q1_lerobot(k: int = 1, episode_length: int | None = None) -> None:
    """在本进程里运行 lerobot 0.3.3 的 eval_main（命令行参数同 lerobot-eval），只做三处运行时替换（不改源码）：
    1) set_seed 前把 torch 后端开关设成与我们脚本相同（eval_main 自己设了 cudnn.benchmark=True）；
    2) rollout 包一层，记录 done 张量算步数、记录实际 GL_RENDERER；
    3) eval_policy 的 max_episodes_rendered 改为 0（不录视频；录视频只是额外渲染，省 CPU 渲染时间）。
    策略由 lerobot 自己从 hub 加载（--policy.path + --policy.n_action_steps），不经过我们的 build_policy。
    episode_length 不为 None 时（--env.episode_length，只截短回合）输出到 q1_lerobot_cpu_k1_len<N>/，
    并把 lerobot rollout 实际发给环境的动作存成 actions.npz，与我们的轨迹逐步逐位比较。
    """
    import lerobot.scripts.eval as le
    import torch

    from chunkre.runner import gl_renderer_of, set_determinism

    out = OUT / ("q1_lerobot_cpu_k1" + (f"_len{episode_length}" if episode_length else ""))
    out.mkdir(parents=True, exist_ok=True)
    captured, info = [], {}
    orig_rollout, orig_seed, orig_eval_policy = le.rollout, le.set_seed, le.eval_policy

    def rollout(*a, **kw):
        env = kw.get("env", a[0] if a else None)
        if "gl" not in info:
            info["gl"] = gl_renderer_of(env)
        r = orig_rollout(*a, **kw)
        captured.append(r["done"].clone())
        info.setdefault("actions", []).append(r["action"].clone())
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
    lout.mkdir(parents=True, exist_ok=True)   # 不录视频时 eval_main 不会自己建这个目录（写 eval_info.json 会失败）
    argv = ["lerobot-eval", f"--policy.path={MODELS['act']['repo']}", f"--policy.n_action_steps={k}",
            "--policy.device=cuda", "--env.type=aloha", "--env.task=AlohaTransferCube-v0",
            f"--eval.batch_size={B}", "--eval.n_episodes=10", "--seed=0", f"--output_dir={lout}"]
    if episode_length:
        argv.append(f"--env.episode_length={episode_length}")
    sys.argv = argv
    try:
        le.init_logging()
        le.eval_main()
    finally:
        le.rollout, le.set_seed, le.eval_policy = orig_rollout, orig_seed, orig_eval_policy
    ev = info["eval"]
    np.savez_compressed(out / "actions.npz", action=torch.cat(info["actions"], dim=0).numpy(),
                        done=torch.cat(captured, dim=0).numpy())
    done = torch.cat(captured, dim=0)
    steps = (torch.argmax(done.to(int), dim=1) + 1).tolist()
    with open(out / "episodes_lerobot.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode_ix", "seed", "success", "steps", "sum_reward", "max_reward"])
        for ep, st in zip(ev["per_episode"], steps):
            w.writerow([ep["episode_ix"], ep["seed"], int(ep["success"]), st, ep["sum_reward"], ep["max_reward"]])
    (out / "config.json").write_text(json.dumps(dict(
        what="lerobot 0.3.3 eval_main（lerobot-eval 同一入口），见 scripts/t3_diag.py q1_lerobot 的三处运行时替换",
        argv=argv, gl_renderer=info.get("gl"), torch_flags=info.get("torch_flags"),
        actions_npz="actions.npz：lerobot rollout 返回的 action (B, T, A)，回合结束后的步被 lerobot 用 0 填充"),
        indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"[Q1 lerobot] pc_success={ev['aggregated']['pc_success']} renderer={info.get('gl')}", flush=True)


# ----------------------------------------------------------------------------------------------
# Q2 / Q4
# ----------------------------------------------------------------------------------------------
def q2() -> None:
    from chunkre.runner import run_eval

    for name, kw in Q2_CONDS:
        frames = "0" if name in Q4_FRAME_CONDS else None
        cfg = load_config(ROOT / "configs" / "act_default.yaml",
                          dict(seeds=Q2_SEEDS, batch_size=B, name=f"q2_{name}", log_traj=True,
                               save_frames=frames, frame_every=10, **kw))
        run_eval(cfg, OUT / f"q2_{name}", log=lambda s: print(s, flush=True))


# ----------------------------------------------------------------------------------------------
# 分析
# ----------------------------------------------------------------------------------------------
def compare(a_rows, b_rows):
    """逐回合比较（除时间列）：返回 (一致回合数, 不一致列计数)。"""
    by = {int(r["seed"]): r for r in b_rows}
    n_same, bad = 0, {}
    for r in a_rows:
        o = by[int(r["seed"])]
        diff = [c for c in r if c not in TIME_COLUMNS and c in o and r[c] != o[c]]
        n_same += not diff
        for c in diff:
            bad[c] = bad.get(c, 0) + 1
    return n_same, bad


def load_traj(run: Path) -> dict:
    zs = [np.load(p) for p in sorted((run / "traj").glob("batch_*.npz"))]
    return zs


def traj_stats(run: Path) -> dict:
    """方块最大抬升、首次夹爪闭合时间步、Q3 距离、每步实际关节移动。"""
    lift, grip_t, grip_init = [], [], []
    dist = {o: [] for o in Q3_OFFSETS}
    dist_arm = {o: [] for o in Q3_OFFSETS}
    step_act, step_pos, late_pos, still_frac, stall_step = [], [], [], [], []
    seeds = []
    for z in load_traj(run):
        alive = z["alive"]
        T, Bn = alive.shape
        cube, pos, act = z["cube_pos"], z["agent_pos"], z["action"]
        for i in range(Bn):
            n = int(z["steps"][i])
            seeds.append(int(z["seeds"][i]))
            lift.append(float(np.nanmax(cube[LIFT_REF_STEP:n, i, 2]) - cube[LIFT_REF_STEP, i, 2]))
            first = None
            for gj in GRIPPERS:
                g = pos[:n, i, gj].astype(np.float64)
                runmax = np.maximum.accumulate(g)
                idx = np.flatnonzero((runmax >= GRIP_OPEN) & (g <= runmax - GRIP_CLOSE_DROP))
                if len(idx) and (first is None or idx[0] < first):
                    first = int(idx[0])
            grip_init.append(float(np.mean(pos[0, i, GRIPPERS])))
            grip_t.append(first)
            step_act.append(float(np.nanmean(np.linalg.norm(np.diff(act[:n, i], axis=0), axis=1))))
            dpos = np.linalg.norm(np.diff(pos[:n, i][:, ARM], axis=0), axis=1)     # 每步手臂关节实际移动
            step_pos.append(float(np.mean(dpos)))
            if n > LATE_STEP + 1:
                late_pos.append(float(np.mean(dpos[LATE_STEP:])))
                still_frac.append(float(np.mean(dpos[LATE_STEP:] < STILL_EPS)))
            # 停滞起点：此后手臂关节一直在回合最后位置 STALL_RADIUS 以内的最早一步（回合跑满 400 步才算）
            if n == T:
                far = np.linalg.norm(pos[:n, i][:, ARM] - pos[n - 1, i][ARM], axis=1) > STALL_RADIUS
                stall_step.append(int(np.flatnonzero(far)[-1] + 1) if far.any() else 0)
        offs = list(z["chunk_offsets"])
        rp = np.flatnonzero(z["repredict"])
        for o in Q3_OFFSETS:
            j = offs.index(o)
            for t in rp:
                ok = alive[t]
                d = z["chunk_sel"][t, ok, j] - pos[t, ok]
                dist[o] += np.linalg.norm(d, axis=1).tolist()
                dist_arm[o] += np.linalg.norm(d[:, ARM], axis=1).tolist()
    closed = [t for t in grip_t if t is not None]
    return dict(
        n=len(lift), mean_max_cube_lift_m=float(np.mean(lift)), max_cube_lift_m=lift,
        gripper_init=float(np.mean(grip_init)),
        n_gripper_closed=len(closed), mean_first_grip_close_step=float(np.mean(closed)) if closed else None,
        first_grip_close_step=grip_t, seeds=seeds,
        mean_step_action_change=float(np.mean(step_act)), mean_step_qpos_change=float(np.mean(step_pos)),
        mean_step_qpos_change_late=float(np.mean(late_pos)) if late_pos else None,
        frac_still_steps_late=float(np.mean(still_frac)) if still_frac else None,
        n_full_length=len(stall_step), stall_step=stall_step,
        median_stall_step=float(np.median(stall_step)) if stall_step else None,
        q3_dist_all14={o + 1: float(np.mean(v)) for o, v in dist.items()},
        q3_dist_arm12={o + 1: float(np.mean(v)) for o, v in dist_arm.items()},
        q3_n_chunks=len(dist[0]),
    )


def analyze() -> None:
    res = {}
    # ---- Q1
    ours = OUT / "q1_ours_cpu_k1"
    ours_t = OUT / "q1_ours_cpu_k1_traj"
    le_csv = OUT / "q1_lerobot_cpu_k1" / "episodes_lerobot.csv"
    q1 = {}
    if (ours / "episodes.csv").exists() and le_csv.exists():
        o = read_episodes(ours / "episodes.csv")
        with open(le_csv, newline="", encoding="utf-8") as f:
            L = {int(r["seed"]): r for r in csv.DictReader(f)}
        per = [dict(seed=int(r["seed"]), ours=(int(r["success"]), int(r["steps"])),
                    lerobot=(int(L[int(r["seed"])]["success"]), int(L[int(r["seed"])]["steps"])),
                    ours_sum_reward=float(r["sum_reward"]), lerobot_sum_reward=float(L[int(r["seed"])]["sum_reward"]))
               for r in o]
        q1.update(
            n=len(per), success_agree=sum(p["ours"][0] == p["lerobot"][0] for p in per),
            steps_agree=sum(p["ours"][1] == p["lerobot"][1] for p in per),
            both_agree=sum(p["ours"] == p["lerobot"] for p in per),
            ours_success=sum(p["ours"][0] for p in per), lerobot_success=sum(p["lerobot"][0] for p in per),
            per_seed=per,
            renderers=dict(
                ours=json.loads((ours / "config.json").read_text(encoding="utf-8"))["gl_renderer"],
                lerobot=json.loads((OUT / "q1_lerobot_cpu_k1" / "config.json").read_text(encoding="utf-8"))
                ["gl_renderer"]))
    if (ours / "episodes.csv").exists() and (ours_t / "episodes.csv").exists():
        a, b = read_episodes(ours / "episodes.csv"), read_episodes(ours_t / "episodes.csv")
        n_same, bad = compare(a, b)
        q1["log_traj_on_off"] = dict(n=len(a), identical_episodes=n_same, mismatched_columns=bad,
                                     traj_files=len(list((ours_t / "traj").glob("*.npz"))),
                                     frames=sorted(p.name for p in (ours_t / "frames").glob("*.gif")))
    # Q1 加强：lerobot 前 100 步实际发出的动作 vs 我们 --log-traj 记录的动作（同一 CPU 渲染，应逐位相同）
    la = OUT / "q1_lerobot_cpu_k1_len100" / "actions.npz"
    tz = sorted((ours_t / "traj").glob("batch_*.npz")) if ours_t.exists() else []
    if la.exists() and tz:
        L = np.load(la)["action"]                       # (B, T, A)，按种子 0..9 顺序
        z = np.load(tz[0])
        O = np.transpose(z["action"], (1, 0, 2))[:, :L.shape[1]]   # (B, T, A)
        same = np.array([np.array_equal(O[i], L[i].astype(O.dtype)) for i in range(len(O))])
        q1["actions_first100"] = dict(
            n_steps=int(L.shape[1]), n_envs=int(len(O)), envs_bitwise_identical=int(same.sum()),
            max_abs_diff=float(np.max(np.abs(O - L))), seeds=[int(x) for x in z["seeds"]])
    res["Q1"] = q1
    # ---- Q2 / Q3 / Q4
    q2r = {}
    for name, _ in Q2_CONDS:
        run = OUT / f"q2_{name}"
        if not (run / "summary.json").exists():
            continue
        m = json.loads((run / "summary.json").read_text(encoding="utf-8"))["metrics"]
        rows = read_episodes(run / "episodes.csv")
        d = dict(n=m["n_episodes"], n_success=m["n_success"], success_rate=m["success_rate"],
                 mean_policy_calls=m["mean_policy_calls"], mean_smoothness=m["mean_smoothness"],
                 mean_path_length=m["mean_path_length"], mean_jerk=m["mean_jerk"],
                 mean_reversal_rate=m["mean_reversal_rate"],
                 n_reversal_nan=sum(math.isnan(float(r["reversal_rate"])) for r in rows),
                 mean_steps=m["mean_steps"], mean_max_reward=m["mean_max_reward"],
                 max_reward_hist={v: sum(float(r["max_reward"]) == v for r in rows) for v in (0, 1, 2, 3, 4)})
        if (run / "traj").exists():
            d.update(traj_stats(run))
        gifs = sorted((run / "frames").glob("*.gif")) if (run / "frames").exists() else []
        d["gifs"] = {p.name: p.stat().st_size for p in gifs}
        q2r[name] = d
    res["Q2"] = q2r
    (OUT / "diag.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(res)


def write_md(res):
    L = ["# T3 ACT k=1 诊断（scripts/t3_diag.py analyze 自动生成）", ""]
    q1 = res.get("Q1", {})
    if "n" in q1:
        L += ["## Q1 评估脚本在 k=1 下是否忠实（CPU 渲染 llvmpipe，种子 0–9，B=10）", "",
              f"- 渲染器：ours `{q1['renderers']['ours']}`；lerobot `{q1['renderers']['lerobot']}`",
              f"- 逐回合一致（成功+步数）：**{q1['both_agree']}/{q1['n']}**（成功 {q1['success_agree']}/{q1['n']}，"
              f"步数 {q1['steps_agree']}/{q1['n']}）",
              f"- 成功：ours {q1['ours_success']}/{q1['n']}，lerobot {q1['lerobot_success']}/{q1['n']}", "",
              "| 种子 | ours (成功, 步数) | lerobot (成功, 步数) | ours sum_reward | lerobot sum_reward |",
              "|---|---|---|---|---|"]
        L += [f"| {p['seed']} | {tuple(p['ours'])} | {tuple(p['lerobot'])} | {p['ours_sum_reward']:g} | "
              f"{p['lerobot_sum_reward']:g} |" for p in q1["per_seed"]]
        L.append("")
    if "actions_first100" in q1:
        a = q1["actions_first100"]
        L += [f"- 加强检查：lerobot（--env.episode_length={a['n_steps']}）实际发出的动作 vs 我们记录的动作，前 {a['n_steps']} 步逐位相同的环境 "
              f"**{a['envs_bitwise_identical']}/{a['n_envs']}**，最大绝对差 {a['max_abs_diff']:.3g}", ""]
    if "log_traj_on_off" in q1:
        t = q1["log_traj_on_off"]
        L += [f"- `--log-traj --save-frames 0` 开/关（CPU 渲染，k=1，种子 0–9）：逐回合（除墙钟列）完全一致 "
              f"**{t['identical_episodes']}/{t['n']}**；不一致列 {t['mismatched_columns'] or '无'}；"
              f"轨迹文件 {t['traj_files']} 个，GIF {t['frames']}", ""]
    q2 = res.get("Q2", {})
    if q2:
        L += ["## Q2 剂量反应（GPU 渲染，种子 0–19，B=10，--log-traj）", "",
              "| 条件 | 成功 | path_length | jerk | reversal_rate | smoothness | 每步关节实际移动 | 方块最大抬升 (m) | "
              "首次夹爪闭合步（闭合回合数） |", "|---|---|---|---|---|---|---|---|---|"]
        for name, d in q2.items():
            g = (f"{d['mean_first_grip_close_step']:.0f}（{d['n_gripper_closed']}/{d['n']}）"
                 if d.get("mean_first_grip_close_step") is not None else f"—（0/{d['n']}）")
            L.append(f"| {name} | {d['n_success']}/{d['n']} | {d['mean_path_length']:.3f} | {d['mean_jerk']:.5f} | "
                     f"{d['mean_reversal_rate']:.3f} | {d['mean_smoothness']:.5f} | "
                     f"{d.get('mean_step_qpos_change', float('nan')):.5f} | "
                     f"{d.get('mean_max_cube_lift_m', float('nan')):.4f} | {g} |")
        L += ["", "## Q3 新预测动作块里第 j 个动作与当时关节位置的平均 L2 距离", "",
              "（arm12 = 12 个手臂关节，弧度；all14 = 再加两个夹爪的归一化开合度）", "",
              "| 条件 | 平均样本数 | j=1 | j=5 | j=10 | j=20 | j=1 (arm12) | j=5 (arm12) | j=10 (arm12) | j=20 (arm12) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for name, d in q2.items():
            if "q3_dist_all14" not in d or name == "te":
                continue
            a, b = d["q3_dist_all14"], d["q3_dist_arm12"]
            L.append(f"| {name} | {d['q3_n_chunks']} | " + " | ".join(f"{a[j]:.4f}" for j in (1, 5, 10, 20)) + " | "
                     + " | ".join(f"{b[j]:.4f}" for j in (1, 5, 10, 20)) + " |")
        L += ["", "## Q4 画面", ""]
        for name, d in q2.items():
            for g, sz in d.get("gifs", {}).items():
                L.append(f"- `q2_{name}/frames/{g}`（{sz / 1024:.0f} KB）")
    (OUT / "diag.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["q1-lerobot", "q2", "analyze"])
    ap.add_argument("--episode-length", type=int, help="q1-lerobot：截短回合（只用于逐步动作比较）")
    a = ap.parse_args()
    if a.what == "q1-lerobot":
        q1_lerobot(episode_length=a.episode_length)
    else:
        {"q2": q2, "analyze": analyze}[a.what]()


if __name__ == "__main__":
    main()
