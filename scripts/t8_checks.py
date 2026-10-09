"""T8b PushT 推 T 形块：基线统计 + D1–D3 检查（结果 results/t8_dp_baseline/、results/t8_checks/；一键 scripts/t8_checks.sh）。

子命令（WSL 内，先 source scripts/env.sh；或 bash scripts/t8_py.sh <子命令>）：
  baseline  读 results/t8_dp_baseline/run_k8（DP k=8，种子 0–49，B=50，--log-traj），统计：
            智能体首次碰到 T 形块的步数、T 形块首次被推动的步数、成功回合的完成步数、各步“已完成”回合的比例；
            从 gym_pusht / pymunk 读 T 形块尺寸、工作区（墙）、智能体半径、目标位姿、reset 的随机范围。
            -> baseline.json、baseline.md、fig_baseline.png
  analyze   读 D1/D2/D3 的运行结果 -> results/t8_checks/checks.json、checks.md、D2 推前推后 PNG

D1 强度 0 = 无扰动：种子 0–49，B=50，k=8：push d=0（d1_d0）vs 基线 run_k8（不开扰动）逐回合一致
   （sum_reward、max_reward 允许 1e-9 以内的浮点差，其他非墙钟、非扰动列必须完全相同），整回合动作逐位相同。
   对照 d1_none_rerun：同一无扰动配置重跑一次，与 run_k8 比较（奖励列同样有 1 ulp 级差异 -> 与扰动钩子无关）。
T8c：重叠时用同一回合的 Generator 重抽方向（最多 20 次）；D2 增加重抽次数分布和方向检查，另有新旧（v1_skip）对照。
D2 扰动生效：d2_k8_dmax（k=8，d=最大档，种子 0–49，--log-traj，种子 D2_FRAME_SEEDS 每步存画面）：
   推的那一步 block_pose_after − block_pose 的长度 = push_d_actual（截断的 < d，其余 = d），角度不变，其他步不动；
   被跳过（skipped_overlap）的回合位姿不变。导出 2 个回合推前 / 推后 PNG。
D3 粗看效果：k ∈ {4, 15}，none vs d=最大档，种子 0–49，报成功率和平均 max_reward。只看方向。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunkre.metrics import TIME_COLUMNS, read_episodes, wilson_ci  # noqa: E402
from chunkre.perturb import (PERTURB_COLUMNS, PushTee, episode_rng, polys_at, ray_clip, tee_agent_gap,  # noqa: E402
                             tee_origin_rect)

BASE = ROOT / "results" / "t8_dp_baseline"
OUT = ROOT / "results" / "t8_checks"
V1 = OUT / "v1_skip"                 # T8b 旧版（重叠直接跳过）的检查结果，T8c 起留档对照
D_LEVELS = (20.0, 40.0, 60.0)      # 建议档位（PushT 坐标），依据见 results/t8_dp_baseline/baseline.md
D_MAX = D_LEVELS[-1]
D2_FRAME_SEEDS = (0, 1, 2, 3)      # 存画面的种子；PNG 从中取前 2 个真正推了的回合
CONTACT_TOL = 0.5                  # 间隙 <= 0.5 视为“碰到”（pymunk 接触有亚像素穿透）
MOVE_TOL_XY, MOVE_TOL_ANG = 1.0, 0.01   # T 形块相对第 0 步位移 > 1 单位或转角 > 0.01 rad 视为“被推动”
REWARD_FLOAT_COLUMNS = ("sum_reward", "max_reward")
REWARD_TOL = 1e-9   # PushT 奖励有 1e-16（max）/ 1e-14（sum）级的浮点差，同一配置重跑也有（D1 对照）


def _load_traj(run: Path) -> list:
    return [dict(np.load(p)) for p in sorted((run / "traj").glob("batch_*.npz"))]


def _pct(a, ps=(0, 5, 10, 25, 50, 75, 90, 100)) -> dict:
    a = np.asarray(a, dtype=float)
    return {str(p): (float(np.percentile(a, p)) if len(a) else None) for p in ps}


# ----------------------------------------------------------------------------------------------
# baseline
# ----------------------------------------------------------------------------------------------
def pusht_geometry() -> dict:
    """建一个 PushT 环境（lerobot 0.3.3 的 pusht 配置同款参数），reset 后从 pymunk 读几何。"""
    import gym_pusht  # noqa: F401
    import gymnasium as gym

    from chunkre.perturb import TeeSim

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    env.reset(seed=0)
    u = env.unwrapped
    sim = TeeSim(u)
    lp = sim.local_polys
    allv = np.vstack(lp)
    areas = [0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))) for p in lp]
    cog = u.block.center_of_gravity
    env.close()
    return dict(
        units="PushT 坐标（画布 512×512；策略看到的图像 96×96，1 个观测像素 = 512/96 = 5.33 单位）",
        tee_rects_local=[p.tolist() for p in lp],
        tee_bbox_local=[float(allv[:, 0].min()), float(allv[:, 0].max()), float(allv[:, 1].min()), float(allv[:, 1].max())],
        tee_bar_w_h=[120.0, 30.0], tee_stem_w_h=[30.0, 90.0], tee_bar_width=30.0,
        tee_area=float(sum(areas)), tee_center_of_gravity_local=[float(cog.x), float(cog.y)],
        workspace_inner=list(sim.workspace), agent_radius=sim.agent_radius,
        goal_pose=[float(x) for x in u.goal_pose],
        reset_ranges=dict(agent_xy="[50, 450)", block_xy="[100, 400)", block_angle="[-π, π)"),
        success="coverage > 0.95（T 形块与目标区域的重叠面积 / 目标面积）；reward = clip(coverage/0.95, 0, 1)",
        control="10 Hz，每步 PD 控制智能体（运动学圆）趋向动作 xy，物理 dt=0.01（每步 10 个子步）",
    )


def baseline() -> None:
    run = BASE / "run_k8"
    geo = pusht_geometry()
    lp = [np.array(p) for p in geo["tee_rects_local"]]
    R = geo["agent_radius"]
    rows = {int(r["seed"]): r for r in read_episodes(run / "episodes.csv")}
    T_env = int(next(iter(rows.values()))["max_steps"])
    per = []
    for z in _load_traj(run):
        bp, ap = z["block_pose"].astype(np.float64), z["agent_pos"].astype(np.float64)
        for i, s in enumerate(z["seeds"]):
            n = int(z["steps"][i])
            contact = move = None
            for t in range(n):
                if contact is None:
                    gap = tee_agent_gap(polys_at(lp, bp[t, i, 2], bp[t, i, :2]), ap[t, i], R)
                    if gap <= CONTACT_TOL:
                        contact = t
                if move is None and (np.linalg.norm(bp[t, i, :2] - bp[0, i, :2]) > MOVE_TOL_XY
                                     or abs(bp[t, i, 2] - bp[0, i, 2]) > MOVE_TOL_ANG):
                    move = t
                if contact is not None and move is not None:
                    break
            g0 = tee_agent_gap(polys_at(lp, bp[0, i, 2], bp[0, i, :2]), ap[0, i], R)
            r = rows[int(s)]
            per.append(dict(seed=int(s), success=int(r["success"]), steps=n, first_contact_step=contact,
                            first_move_step=move, gap_at_step0=round(float(g0), 3),
                            max_reward=float(r["max_reward"])))
    per.sort(key=lambda d: d["seed"])
    succ_steps = [d["steps"] for d in per if d["success"]]
    contact = [d["first_contact_step"] for d in per if d["first_contact_step"] is not None]
    move = [d["first_move_step"] for d in per if d["first_move_step"] is not None]
    n = len(per)
    # 第 t 步 on_state 时“已经成功结束”的回合 = 完成步数 steps <= t（结束那一步的 env.step 发生在第 steps-1 步）
    done_by = {t: int(sum(1 for x in succ_steps if x <= t)) for t in (10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150)}
    res = dict(
        source=str(run.relative_to(ROOT)), n=n, n_success=len(succ_steps), max_steps=T_env,
        geometry=geo,
        first_contact_step_pct=_pct(contact), n_contact=len(contact),
        first_move_step_pct=_pct(move), n_move=len(move),
        success_steps_pct=_pct(succ_steps),
        n_done_by_step=done_by,
        n_contact_le=dict((t, int(sum(1 for x in contact if x <= t))) for t in (5, 10, 15, 20, 30, 40, 60)),
        definitions=dict(
            first_contact_step=f"T 形块与智能体（圆，半径 {R:g}）间隙 <= {CONTACT_TOL} 的第一步（轨迹第 t 步 = 执行动作前）",
            first_move_step=f"T 形块 body 位置相对第 0 步移动 > {MOVE_TOL_XY} 或转角 > {MOVE_TOL_ANG} rad 的第一步",
            success_steps="成功回合的 steps（含结束那一步）",
            n_done_by_step="steps <= t 的成功回合数 = 第 t 步推的时候已经结束（不会被推）的回合",
        ),
        per_episode=per,
    )
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "baseline.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (BASE / "baseline.md").write_text(baseline_md(res), encoding="utf-8", newline="\n")
    fig_baseline(res)
    print(baseline_md(res))


def baseline_md(r: dict) -> str:
    g = r["geometry"]
    f = lambda d: " / ".join(f"{k}%:{v:g}" for k, v in d.items() if v is not None)  # noqa: E731
    L = [f"# T8b DP 基线（{r['source']}，k=8，种子 0–49，B=50）", "",
         f"- 成功 {r['n_success']}/{r['n']}；回合上限 {r['max_steps']} 步（10 Hz）。",
         f"- 智能体首次碰到 T 形块（{r['n_contact']}/{r['n']} 回合碰到）步数分位：{f(r['first_contact_step_pct'])}",
         f"- T 形块首次被推动（{r['n_move']}/{r['n']}）步数分位：{f(r['first_move_step_pct'])}",
         f"- 成功回合完成步数分位：{f(r['success_steps_pct'])}",
         "- 第 t 步之前已经成功结束的回合数（推的时候已经不会被推）：" +
         "，".join(f"t={t}: {v}" for t, v in r["n_done_by_step"].items()),
         "- 第 t 步前已经碰到 T 形块的回合数：" + "，".join(f"≤{t}: {v}" for t, v in r["n_contact_le"].items()),
         "", "## 几何（gym_pusht / pymunk 读出）", "",
         f"- 单位：{g['units']}",
         f"- T 形块：横条 {g['tee_bar_w_h'][0]:g}×{g['tee_bar_w_h'][1]:g}，竖条 {g['tee_stem_w_h'][0]:g}×{g['tee_stem_w_h'][1]:g}；"
         f"整体包围盒 {g['tee_bbox_local'][1] - g['tee_bbox_local'][0]:g}×{g['tee_bbox_local'][3] - g['tee_bbox_local'][2]:g}；"
         f"面积 {g['tee_area']:g}；body 原点 = 横条一条长边的中点，重心（局部）{g['tee_center_of_gravity_local']}",
         f"- 工作区（墙内缘）：x、y ∈ [{g['workspace_inner'][0]:g}, {g['workspace_inner'][1]:g}]；智能体半径 {g['agent_radius']:g}",
         f"- 目标位姿：{g['goal_pose']}（x, y, 角度）；reset：{g['reset_ranges']}",
         f"- 成功：{g['success']}", f"- 控制：{g['control']}", ""]
    return "\n".join(L)


def fig_baseline(r: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = r["per_episode"]
    T = r["max_steps"]
    ts = np.arange(T + 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for key, lab, col in [("first_contact_step", "agent first touches T", "#0072B2"),
                          ("first_move_step", "T first moves", "#009E73")]:
        v = np.array([d[key] for d in per if d[key] is not None])
        ax.step(ts, [(v <= t).mean() * len(v) / len(per) for t in ts], where="post", color=col, label=lab)
    sv = np.array([d["steps"] for d in per if d["success"]])
    ax.step(ts, [(sv <= t).sum() / len(per) for t in ts], where="post", color="#D55E00", label="episode succeeded")
    ax.set_xlabel("step (10 Hz)")
    ax.set_ylabel(f"fraction of {len(per)} episodes")
    ax.set_title("T8b baseline: DP k=8, PushT, seeds 0-49", fontsize=10)
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.02)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(BASE / "fig_baseline.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# analyze
# ----------------------------------------------------------------------------------------------
def _compare_runs(a: Path, b: Path) -> dict:
    """逐回合比较两次运行：奖励列（REWARD_FLOAT_COLUMNS）允许 REWARD_TOL 以内的浮点差，其他非墙钟、非扰动列必须字符串完全相同；
    两边都有轨迹时再比整回合执行动作是否逐位相同。"""
    ra, rb = read_episodes(a / "episodes.csv"), read_episodes(b / "episodes.csv")
    skip = set(TIME_COLUMNS) | set(PERTURB_COLUMNS) | set(REWARD_FLOAT_COLUMNS)
    cols = [c for c in ra[0] if c not in skip]
    key = lambda r: int(r["seed"])  # noqa: E731
    ra, rb = sorted(ra, key=key), sorted(rb, key=key)
    assert [key(r) for r in ra] == [key(r) for r in rb]
    same_ep = 0
    diff_cols, exact_diff = set(), {c: 0 for c in REWARD_FLOAT_COLUMNS}
    max_diff = {c: 0.0 for c in REWARD_FLOAT_COLUMNS}
    for x, y in zip(ra, rb):
        bad = [c for c in cols if x[c] != y[c]]
        for c in REWARD_FLOAT_COLUMNS:
            dv = abs(float(x[c]) - float(y[c]))
            max_diff[c] = max(max_diff[c], dv)
            exact_diff[c] += int(x[c] != y[c])
            if dv > REWARD_TOL:
                bad.append(c)
        diff_cols |= set(bad)
        same_ep += int(not bad)
    acts_same = None
    if (a / "traj").exists() and (b / "traj").exists():
        acts_same = 0
        for za, zb in zip(_load_traj(a), _load_traj(b)):
            for i in range(len(za["seeds"])):
                n = int(za["steps"][i])
                acts_same += int(np.array_equal(za["action"][:n, i], zb["action"][:n, i]))
    return dict(a=str(a.relative_to(ROOT)), b=str(b.relative_to(ROOT)), n=len(ra), compared_exact_columns=cols,
                episodes_identical=same_ep, columns_with_diff=sorted(diff_cols),
                reward_max_abs_diff=max_diff, reward_n_not_bitwise_equal=exact_diff,
                full_episode_actions_identical=acts_same,
                success_a=sum(int(r["success"]) for r in ra), success_b=sum(int(r["success"]) for r in rb),
                push_status_b=sorted({r.get("push_status") or "" for r in rb}))


def d1() -> dict:
    main = _compare_runs(BASE / "run_k8", OUT / "d1_d0")
    # 对照：同一配置（不开扰动）重跑一次。说明奖励列的 1 ulp 级差异与扰动钩子无关
    ctrl = _compare_runs(BASE / "run_k8", OUT / "d1_none_rerun") if (OUT / "d1_none_rerun").exists() else None
    passed = (main["episodes_identical"] == main["n"] and main["push_status_b"] == ["zero"]
              and main["full_episode_actions_identical"] == main["n"])
    return dict(main=main, control_none_vs_none=ctrl, passed=passed)

def d2() -> dict:
    run = OUT / "d2_k8_dmax"
    rows = {int(r["seed"]): r for r in read_episodes(run / "episodes.csv")}
    n_pushed = n_clip = n_skip = n_nr = 0
    max_err = max_ang = max_other = max_dir = 0.0
    redraw_hist: dict[int, int] = {}
    bad = []
    for z in _load_traj(run):
        bp, ba = z["block_pose"].astype(np.float64), z["block_pose_after"].astype(np.float64)
        for i, s in enumerate(z["seeds"]):
            r = rows[int(s)]
            n = int(z["steps"][i])
            delta = ba[:n, i] - bp[:n, i]
            moved = np.flatnonzero(np.abs(delta).max(axis=1) > 0)
            st = r["push_status"]
            if st != "not_reached":
                nr_ = int(r["push_redraws"])
                redraw_hist[nr_] = redraw_hist.get(nr_, 0) + 1
            if st == "pushed":
                t = int(r["push_step"])
                n_pushed += 1
                d_act = float(r["push_d_actual"])
                n_clip += int(r["push_clip"] == "wall")
                expect = D_MAX if r["push_clip"] == "" else d_act
                L = float(np.linalg.norm(delta[t, :2]))
                # 轨迹是 float32：位置 ~300 时分辨率约 3e-5
                err = abs(L - expect) if r["push_clip"] == "" else abs(L - d_act)
                max_err = max(max_err, err)
                max_ang = max(max_ang, abs(delta[t, 2]))
                if L > 1.0:       # 位移方向 = 最终（重抽后）方向 push_angle_deg
                    dth = math.atan2(delta[t, 1], delta[t, 0]) - math.radians(float(r["push_angle_deg"]))
                    max_dir = max(max_dir, abs(math.atan2(math.sin(dth), math.cos(dth))))
                others = [m for m in moved if m != t]
                if others:
                    max_other = max(max_other, float(np.abs(delta[others]).max()))
                if err > 1e-3 or abs(delta[t, 2]) > 0 or others or (r["push_clip"] == "wall" and not d_act < D_MAX):
                    bad.append(int(s))
            else:
                n_skip += int(st == "skipped_overlap")
                n_nr += int(st == "not_reached")
                if len(moved):
                    bad.append(int(s))
    return dict(run=str(run.relative_to(ROOT)), d=D_MAX, n=len(rows), n_pushed=n_pushed, n_clipped_wall=n_clip,
                n_skipped_overlap=n_skip, n_not_reached=n_nr, max_abs_len_error=max_err,
                max_abs_angle_change=max_ang, max_change_at_other_steps=max_other, bad_seeds=bad,
                max_direction_error_rad=max_dir, redraw_hist=dict(sorted(redraw_hist.items())),
                passed=not bad and n_pushed > 0 and max_dir < 1e-3, pngs=d2_pngs(run, rows))


def d2_pngs(run: Path, rows: dict) -> list[str]:
    """推前（第 t 步观测）/ 推后（第 t+1 步观测）各 1 帧，放大 4 倍拼在一起（96×96 -> 384×384，最近邻）。"""
    from PIL import Image, ImageDraw

    out = []
    for s in D2_FRAME_SEEDS:
        r = rows[s]
        if r["push_status"] != "pushed" or float(r["push_d_actual"]) <= 0:
            continue
        t = int(r["push_step"])
        fa, fb = run / "frames" / f"seed{s}" / f"step{t:04d}.png", run / "frames" / f"seed{s}" / f"step{t + 1:04d}.png"
        if not (fa.exists() and fb.exists()):
            continue
        ims = [Image.open(p).convert("RGB").resize((384, 384), Image.NEAREST) for p in (fa, fb)]
        W = Image.new("RGB", (384 * 2 + 12, 384 + 28), "white")
        W.paste(ims[0], (0, 28))
        W.paste(ims[1], (396, 28))
        dr = ImageDraw.Draw(W)
        dr.text((4, 6), f"seed {s}: step {t} (before push)", fill="black")
        dr.text((400, 6), f"step {t + 1} (after push d={float(r['push_d_actual']):.1f}, "
                          f"{float(r['push_angle_deg']):.0f} deg)", fill="black")
        p = OUT / f"d2_seed{s}_step{t}.png"
        W.save(p)
        out.append(str(p.relative_to(ROOT)))
        if len(out) == 2:
            break
    return out


def _replay(seed, pre, agent, ws, lp, R, d, max_redraws=20):
    """离线重放 PushTee._push 的方向抽取：episode_rng 先抽 push_step、θ，重叠时继续抽 θ，最多 max_redraws 次。
    返回 (skipped, 撞墙截断, 重抽次数)。"""
    g = episode_rng(seed, "push")
    g.integers(PushTee.DEFAULTS["t_lo"], PushTee.DEFAULTS["t_hi"] + 1)
    th = float(g.uniform(0.0, 2.0 * math.pi))
    polys = polys_at(lp, pre[2], pre[:2])
    rect = tee_origin_rect(polys, pre[:2], ws)
    n = 0
    while True:
        u = np.array([math.cos(th), math.sin(th)])
        st = ray_clip(pre[:2], u, d, rect)
        gap = tee_agent_gap(polys_at(lp, pre[2], pre[:2] + st * u), agent, R)
        if gap >= 0 or n >= max_redraws:
            return gap < 0, st < d - 1e-9, n
        th = float(g.uniform(0.0, 2.0 * math.pi))
        n += 1


def skip_estimate() -> dict:
    """离线估算三档 d 下 k=8 的截断 / 跳过 / 重抽数：同一种子、同一 k，推之前的轨迹与 d 无关（推的时刻和方向也只由种子决定），
    所以用 D2 轨迹里推那一步的 T 形块位姿和智能体位置，对每档 d 重放“抽方向 → 截断 → 重叠检查 → 重抽”。
    d=最大档的 skipped_overlap、重抽次数必须与实际运行逐回合一致。同时给出“不重抽”（= T8b 旧规则）的跳过数。"""
    geo = json.loads((BASE / "baseline.json").read_text(encoding="utf-8"))["geometry"]
    lp = [np.array(p) for p in geo["tee_rects_local"]]
    ws, R = geo["workspace_inner"], geo["agent_radius"]
    run = OUT / "d2_k8_dmax"
    rows = {int(r["seed"]): r for r in read_episodes(run / "episodes.csv")}
    res, mismatch = {}, []
    for d in D_LEVELS:
        sk = cl = n_contact = sk_contact = sk_v1 = n_redrawn = 0
        hist: dict[int, int] = {}
        for z in _load_traj(run):
            for i, s in enumerate(z["seeds"]):
                r = rows[int(s)]
                if r["push_status"] == "not_reached":
                    continue
                t = int(r["push_step"])
                bp, ap = z["block_pose"][t, i].astype(np.float64), z["agent_pos"][t, i].astype(np.float64)
                skip, clip, n = _replay(int(s), bp, ap, ws, lp, R, d)
                skip1, _, _ = _replay(int(s), bp, ap, ws, lp, R, d, max_redraws=0)
                contact = tee_agent_gap(polys_at(lp, bp[2], bp[:2]), ap, R) <= CONTACT_TOL
                cl += int(clip)
                sk += int(skip)
                sk_v1 += int(skip1)
                n_redrawn += int(n > 0)
                hist[n] = hist.get(n, 0) + 1
                n_contact += int(contact)
                sk_contact += int(skip and contact)
                if d == D_MAX and (skip != (r["push_status"] == "skipped_overlap") or n != int(r["push_redraws"])):
                    mismatch.append(int(s))
        res[f"d{d:g}"] = dict(skipped_overlap=sk, skipped_if_no_redraw=sk_v1, redrawn=n_redrawn,
                              redraw_hist=dict(sorted(hist.items())), clipped_wall=cl,
                              in_contact_at_push=n_contact, skipped_while_in_contact=sk_contact)
    return dict(run=str(run.relative_to(ROOT)), n=len(rows), levels=res, mismatch_vs_actual_at_dmax=mismatch,
                note="轨迹为 float32，重放只作估算；d=最大档与实际运行逐回合比对（skipped_overlap 和重抽次数）")


def d3() -> dict:
    res = {}
    for k in (4, 15):
        for cond in ("none", "dmax"):
            p = OUT / f"d3_k{k}_{cond}" / "episodes.csv"
            if not p.exists():
                continue
            rows = read_episodes(p)
            ns = sum(int(r["success"]) for r in rows)
            lo, hi = wilson_ci(ns, len(rows))
            st = {}
            for r in rows:
                st[r["push_status"] or "none"] = st.get(r["push_status"] or "none", 0) + 1
            res[f"k{k}_{cond}"] = dict(n=len(rows), n_success=ns, success_rate=ns / len(rows), ci=[lo, hi],
                                       mean_max_reward=float(np.mean([float(r["max_reward"]) for r in rows])),
                                       push_status=st)
    # k=8 的两格：基线和 D2
    for name, p in (("k8_none", BASE / "run_k8"), ("k8_dmax", OUT / "d2_k8_dmax")):
        rows = read_episodes(p / "episodes.csv")
        ns = sum(int(r["success"]) for r in rows)
        st = {}
        for r in rows:
            st[r["push_status"] or "none"] = st.get(r["push_status"] or "none", 0) + 1
        res[name] = dict(n=len(rows), n_success=ns, success_rate=ns / len(rows), ci=list(wilson_ci(ns, len(rows))),
                         mean_max_reward=float(np.mean([float(r["max_reward"]) for r in rows])), push_status=st)
    return res


def analyze() -> None:
    out = dict(d1=d1(), d2=d2(), skip_estimate=skip_estimate(), d3=d3())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checks.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    md = checks_md(out)
    (OUT / "checks.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)


def v1_compare_md(o: dict) -> list[str]:
    """T8b 旧版（重叠直接跳过，results/t8_checks/v1_skip）与新版（重抽）的实际运行数字对照。"""
    f = V1 / "checks.json"
    if not f.exists():
        return []
    v1 = json.loads(f.read_text(encoding="utf-8"))
    L = ["### 新旧对照（实际运行，d=60，种子 0–49）", "",
         "| k | 旧 skipped_overlap | 新 skipped_overlap | 旧推了 | 新推了 | 旧成功 | 新成功 |", "|---|---|---|---|---|---|---|"]
    for k in (4, 8, 15):
        name = f"k{k}_dmax"
        if name in v1["d3"] and name in o["d3"]:
            a, b = v1["d3"][name], o["d3"][name]
            L.append(f"| {k} | {a['push_status'].get('skipped_overlap', 0)} | {b['push_status'].get('skipped_overlap', 0)} | "
                     f"{a['push_status'].get('pushed', 0)} | {b['push_status'].get('pushed', 0)} | "
                     f"{a['n_success']}/{a['n']} | {b['n_success']}/{b['n']} |")
    L.append("")
    return L


def checks_md(o: dict) -> str:
    a, b, c = o["d1"]["main"], o["d2"], o["d3"]
    ctl = o["d1"]["control_none_vs_none"]
    def _rw(x):
        return "，".join(f"{k} 最大差 {x['reward_max_abs_diff'][k]:.2g}（不逐位相等 {x['reward_n_not_bitwise_equal'][k]} 回合）"
                        for k in REWARD_FLOAT_COLUMNS)

    L = ["# T8b 检查 D1–D3（PushT 推 T 形块，DP）", "",
         f"## D1 强度 0 = 无扰动 —— {'通过' if o['d1']['passed'] else '未通过'}", "",
         f"- {a['a']}（不开扰动）vs {a['b']}（push d=0），种子 0–49，B=50，k=8",
         f"- 逐回合一致 {a['episodes_identical']}/{a['n']}（除墙钟列、扰动列外，其他列字符串完全相同；"
         f"sum_reward、max_reward 允许 ≤ {REWARD_TOL:g}）；超出的列：{a['columns_with_diff'] or '无'}",
         f"- 奖励列：{_rw(a)}",
         f"- d=0 运行的 push_status：{a['push_status_b']}；整回合执行动作逐位相同："
         f"{a['full_episode_actions_identical']}/{a['n']}；成功 {a['success_a']}/{a['n']} vs {a['success_b']}/{a['n']}",
    ]
    if ctl:
        L += [f"- 对照（同一无扰动配置重跑，{ctl['b']}）：逐回合一致 {ctl['episodes_identical']}/{ctl['n']}，"
              f"整回合动作逐位相同 {ctl['full_episode_actions_identical']}/{ctl['n']}；{_rw(ctl)}。"
              "即奖励列的 1 ulp 级差异在不开扰动时也有（PushT 环境自身的浮点不确定性，来源未查），与扰动钩子无关。"]
    L += ["",
         f"## D2 扰动生效（d={b['d']:g}，k=8，种子 0–49）—— {'通过' if b['passed'] else '未通过'}", "",
         f"- 推了 {b['n_pushed']}（其中撞墙截断 {b['n_clipped_wall']}），skipped_overlap {b['n_skipped_overlap']}，"
         f"not_reached {b['n_not_reached']}（共 {b['n']}）",
         f"- 推的那一步位移长度与期望（未截断 = d，截断 = push_d_actual）的最大误差 {b['max_abs_len_error']:.2e}"
         f"（轨迹为 float32）；最大转角变化 {b['max_abs_angle_change']:g}；其他步 after−before 最大 {b['max_change_at_other_steps']:g}",
         f"- 重抽次数分布（推了或被跳过的回合，共 {sum(b['redraw_hist'].values())} 个）：{b['redraw_hist']}；"
         f"推后位移方向与记录的最终方向 push_angle_deg 的最大差 {b['max_direction_error_rad']:.2e} rad",
         f"- 不合格种子：{b['bad_seeds'] or '无'}",
         "- 推前推后对比图：" + "、".join(f"`{p}`" for p in b["pngs"]), "",
         "### 三档 d 的截断 / 跳过 / 重抽数（k=8，用 D2 轨迹离线重放；d=最大档的跳过和重抽次数与实际运行逐回合一致："
         f"{'是' if not o['skip_estimate']['mismatch_vs_actual_at_dmax'] else '否 ' + str(o['skip_estimate']['mismatch_vs_actual_at_dmax'])}）", "",
         "| d | skipped_overlap（新，重抽后） | 若不重抽（= 旧规则）的 skipped_overlap | 重抽过的回合 | 重抽次数分布 | 撞墙截断 | 推时贴着 T | 其中仍被跳过 |",
         "|---|---|---|---|---|---|---|---|",
         *[f"| {k[1:]} | {v['skipped_overlap']}/{o['skip_estimate']['n']} | {v['skipped_if_no_redraw']} | {v['redrawn']} | "
           f"{v['redraw_hist']} | {v['clipped_wall']} | {v['in_contact_at_push']} | {v['skipped_while_in_contact']} |"
           for k, v in o['skip_estimate']['levels'].items()],
         "",
         *v1_compare_md(o),
         "## D3 粗看（只看方向，不作结论）", "",
         "| 格子 | 成功 | 成功率 [Wilson 95%] | 平均 max_reward | 推的状态 |", "|---|---|---|---|---|"]
    for name in ("k4_none", "k4_dmax", "k8_none", "k8_dmax", "k15_none", "k15_dmax"):
        if name in c:
            x = c[name]
            L.append(f"| {name} | {x['n_success']}/{x['n']} | {x['success_rate']:.2f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}] | "
                     f"{x['mean_max_reward']:.3f} | {x.get('push_status', '')} |")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["baseline", "analyze"])
    a = ap.parse_args(argv)
    {"baseline": baseline, "analyze": analyze}[a.cmd]()


if __name__ == "__main__":
    main()
