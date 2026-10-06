"""T4 扰动框架检查（结果 results/t4_checks/；一键运行 scripts/t4_checks.sh）。

子命令：
  window   由 T3 的 k=100 轨迹（results/t3_k1_diag/q2_k100/traj/）统计首次夹爪闭合步数分布 -> window.json
  analyze  读 P1/P2/P3 的运行结果，做比较，写 checks.json / checks.md，导出 P2 推前后画面

P1 强度 0 = 无扰动：CPU 渲染，种子 0–9，B=10，k=100，--log-traj；p1_none（不开扰动）vs p1_d0（push d=0）
   逐回合结果（除墙钟列和扰动列外所有列）完全一致，且前 100 步实际动作逐位相同。
P2 扰动生效：GPU，d=最大档（D_MAX），种子 0–9（取自 P3 的 p3_k100_dmax 第 0 批）：
   轨迹里推的那一步 cube_pos_after - cube_pos 的水平长度 = d（截断的除外，误差 < 1 mm），z 不变，其他步不动；
   导出 2 个回合推前（第 t 步观测）/ 推后（第 t+1 步观测）各 1 帧 PNG。
P3 粗看效果：GPU，种子 0–19，k ∈ {10, 100} + te，d ∈ {0, D_MAX}，只报成功率。
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chunkre.metrics import TIME_COLUMNS, read_episodes, summarize_csv  # noqa: E402
from chunkre.perturb import PERTURB_COLUMNS  # noqa: E402

OUT = ROOT / "results" / "t4_checks"
D_MAX = 0.10
P2_FRAME_SEEDS = (0, 1)
P3_RUNS = [(k, d) for k in ("10", "100", "te") for d in ("d0", "dmax")]
TOL_M = 1e-3


def window() -> None:
    import t3_diag as t3

    run = ROOT / "results" / "t3_k1_diag" / "q2_k100"
    st = t3.traj_stats(run)
    g = np.array([t for t in st["first_grip_close_step"] if t is not None], dtype=float)
    # 方块第一次被碰到：水平位置比第 10 步（落稳后）偏离 > 2 mm 的第一步
    touch = []
    for z in t3.load_traj(run):
        for i in range(z["cube_pos"].shape[1]):
            n = int(z["steps"][i])
            c = z["cube_pos"][:n, i].astype(np.float64)
            moved = np.flatnonzero(np.linalg.norm(c[:, :2] - c[10, :2], axis=1) > 0.002)
            touch.append(int(moved[0]) if len(moved) else None)
    tt = np.array([t for t in touch if t is not None], dtype=float)
    pct = lambda a: {str(p): float(np.percentile(a, p)) for p in (0, 5, 10, 25, 50)}  # noqa: E731
    res = dict(
        source=str(run.relative_to(ROOT)), n=st["n"], n_gripper_closed=int(len(g)),
        first_grip_close_step=st["first_grip_close_step"], first_grip_close_pct=pct(g),
        first_cube_touch_step=touch, first_cube_touch_pct=pct(tt),
        n_touch_before_100=int(np.sum(tt < 100)), n_touch_le_100=int(np.sum(tt <= 100)),
        chosen_window=[20, 100],
        definition="首次夹爪闭合 = t3_diag.traj_stats 的定义；首次碰到 = 方块 xy 偏离第 10 步位置 > 2 mm 的第一步",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "window.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if "pct" in k or k.startswith("n")}, indent=2))


def _load_traj(run: Path) -> list:
    return [dict(np.load(p)) for p in sorted((run / "traj").glob("batch_*.npz"))]


def p1() -> dict:
    a, b = OUT / "p1_none", OUT / "p1_d0"
    ra, rb = read_episodes(a / "episodes.csv"), read_episodes(b / "episodes.csv")
    skip = set(TIME_COLUMNS) | set(PERTURB_COLUMNS)
    cols = [c for c in ra[0] if c not in skip]
    key = lambda r: int(r["seed"])  # noqa: E731
    ra, rb = sorted(ra, key=key), sorted(rb, key=key)
    same_ep = [all(x[c] == y[c] for c in cols) for x, y in zip(ra, rb)]
    ta, tb = _load_traj(a)[0], _load_traj(b)[0]
    acts_same, first_diff = [], []
    for i in range(ta["action"].shape[1]):
        x, y = ta["action"][:100, i], tb["action"][:100, i]
        eq = np.array_equal(x, y, equal_nan=True)
        acts_same.append(bool(eq))
        if not eq:
            first_diff.append(int(np.flatnonzero(~np.all((x == y) | (np.isnan(x) & np.isnan(y)), axis=1))[0]))
    # 整个回合的动作也比较一下（更严）
    full_same = bool(np.array_equal(ta["action"], tb["action"], equal_nan=True))
    obs_same = bool(np.array_equal(ta["agent_pos"], tb["agent_pos"], equal_nan=True)
                    and np.array_equal(ta["cube_pos"], tb["cube_pos"], equal_nan=True))
    cfg_a = json.loads((a / "config.json").read_text(encoding="utf-8"))
    cfg_b = json.loads((b / "config.json").read_text(encoding="utf-8"))
    status = sorted({r["push_status"] for r in rb})
    return dict(
        n=len(ra), compared_columns=cols, n_episodes_identical=int(sum(same_ep)),
        n_envs_first100_actions_bitwise_identical=int(sum(acts_same)), first_diff_steps=first_diff,
        full_episode_actions_bitwise_identical=full_same, agent_pos_and_cube_pos_identical=obs_same,
        success=[int(r["success"]) for r in ra], push_status_d0=status,
        renderer=[cfg_a["gl_renderer"], cfg_b["gl_renderer"]],
        passed=bool(all(same_ep) and all(acts_same) and len(ra) == 10),
    )


def p2() -> dict:
    run = OUT / "p3_k100_dmax"
    z = _load_traj(run)[0]                       # 第 0 批 = 种子 0–9
    rows = {int(r["seed"]): r for r in read_episodes(run / "episodes.csv")}
    cp, ca = z["cube_pos"].astype(np.float64), z["cube_pos_after"].astype(np.float64)
    out, frames = [], []
    for i, s in enumerate(z["seeds"].tolist()):
        r = rows[s]
        disp = np.linalg.norm(np.nan_to_num(ca[:, i, :2] - cp[:, i, :2]), axis=1)
        moved_steps = np.flatnonzero(disp > 1e-9).tolist()
        t = int(r["push_step"])
        d_xy = float(disp[t]) if t < len(disp) else float("nan")
        dz = float(abs(ca[t, i, 2] - cp[t, i, 2])) if t < len(disp) else float("nan")
        # 推后一步（物理走 1 个控制步 = 10 个子步）方块又动了多少
        drift = float(np.linalg.norm(cp[t + 1, i, :2] - ca[t, i, :2])) if t + 1 < len(cp) else float("nan")
        clipped = bool(r["push_clip"])
        err = abs(d_xy - D_MAX)
        csv_d = float(r["push_d_actual"]) if r["push_d_actual"] != "" else float("nan")
        out.append(dict(seed=s, push_step=t, push_angle_deg=float(r["push_angle_deg"]), status=r["push_status"],
                        clip=r["push_clip"], traj_disp_xy=d_xy, traj_dz=dz, csv_d_actual=csv_d,
                        abs_err_vs_d=err, moved_steps=moved_steps, drift_next_step=drift,
                        success=int(r["success"]),
                        ok=bool(r["push_status"] == "pushed" and moved_steps == [t] and dz < 1e-9
                                and abs(d_xy - csv_d) < 1e-6 and (clipped or err < TOL_M))))
    unclipped = [o for o in out if o["status"] == "pushed" and not o["clip"]]
    # 导出画面：推前 = 第 t 步观测（策略选第 t 步动作时看到的），推后 = 第 t+1 步观测
    fdir = OUT / "p2_frames"
    fdir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for s in P2_FRAME_SEEDS:
        t = int(rows[s]["push_step"])
        src = run / "frames" / f"seed{s}"
        pre, post = src / f"step{t:04d}.png", src / f"step{t + 1:04d}.png"
        if pre.exists() and post.exists():
            a, b = fdir / f"seed{s}_step{t:04d}_pre.png", fdir / f"seed{s}_step{t + 1:04d}_post.png"
            shutil.copy(pre, a)
            shutil.copy(post, b)
            ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
            pair = Image.new("RGB", (ia.width + ib.width + 8, ia.height), (255, 255, 255))
            pair.paste(ia, (0, 0))
            pair.paste(ib, (ia.width + 8, 0))
            pair.save(fdir / f"seed{s}_pair.png")
            frames += [str(a.relative_to(ROOT)), str(b.relative_to(ROOT))]
    return dict(
        d=D_MAX, n=len(out), episodes=out, n_ok=int(sum(o["ok"] for o in out)),
        n_unclipped=len(unclipped), n_clipped=int(sum(bool(o["clip"]) for o in out)),
        max_abs_err_unclipped_m=float(max(o["abs_err_vs_d"] for o in unclipped)) if unclipped else None,
        mean_drift_next_step_m=float(np.nanmean([o["drift_next_step"] for o in out])),
        frames=frames,
        passed=bool(len(out) == 10 and all(o["ok"] for o in out) and len(frames) == 4),
    )


def cleanup_frames() -> dict:
    """p3_k100_dmax 为了拿推前后的画面每步都存了 PNG（2 × 400 张）：只留 GIF，其余删掉（需要的两帧已复制到 p2_frames）。"""
    n = 0
    for d in (OUT / "p3_k100_dmax" / "frames").glob("seed*"):
        if d.is_dir():
            n += len(list(d.glob("*.png")))
            shutil.rmtree(d)
    return dict(deleted_pngs=n)


def p3() -> dict:
    res = {}
    for k, d in P3_RUNS:
        run = OUT / f"p3_k{k}_{d}"
        m = summarize_csv(run / "episodes.csv")
        res[f"k{k}_{d}"] = dict(k=k, d=0.0 if d == "d0" else D_MAX, n=m["n_episodes"], n_success=m["n_success"],
                                success_rate=m["success_rate"], ci95=m["success_ci95_wilson"],
                                push_status_counts=m.get("push_status_counts"), n_push_clipped=m.get("n_push_clipped"))
    return res


def write_md(r: dict) -> None:
    L = ["# T4 扰动框架检查（`scripts/t4_checks.sh`，`scripts/t4_checks.py analyze` 生成）", ""]
    a = r["p1"]
    L += [f"## P1 强度 0 = 无扰动（CPU 渲染，种子 0–9，k=100）— {'通过' if a['passed'] else '未通过'}",
          f"- 逐回合结果一致：{a['n_episodes_identical']}/{a['n']}（比较列：除墙钟列和扰动列外全部 {len(a['compared_columns'])} 列）",
          f"- 前 100 步动作逐位相同：{a['n_envs_first100_actions_bitwise_identical']}/{a['n']}；"
          f"整回合动作逐位相同：{a['full_episode_actions_bitwise_identical']}；agent_pos 与方块位置轨迹相同：{a['agent_pos_and_cube_pos_identical']}",
          f"- d=0 回合的 push_status：{a['push_status_d0']}；渲染器：{a['renderer'][0]}", ""]
    b = r["p2"]
    L += [f"## P2 扰动生效（GPU，d={b['d']} m，种子 0–9）— {'通过' if b['passed'] else '未通过'}",
          f"- 检查通过 {b['n_ok']}/{b['n']}；未截断 {b['n_unclipped']} 个，最大误差 {b['max_abs_err_unclipped_m']:.2e} m；截断 {b['n_clipped']} 个",
          f"- 推后 1 个控制步方块又移动的平均距离：{b['mean_drift_next_step_m'] * 1000:.2f} mm",
          "", "| 种子 | 推的步 | 角度° | 状态 | 截断 | 轨迹位移 m | z 变化 | 推后 1 步漂移 mm | 成功 |", "|---|---|---|---|---|---|---|---|---|"]
    for o in b["episodes"]:
        L.append(f"| {o['seed']} | {o['push_step']} | {o['push_angle_deg']:.1f} | {o['status']} | {o['clip'] or '-'} | "
                 f"{o['traj_disp_xy']:.6f} | {o['traj_dz']:.1e} | {o['drift_next_step'] * 1000:.2f} | {o['success']} |")
    L += ["", "画面（推前 = 第 t 步观测，推后 = 第 t+1 步观测）：" + "、".join(f"`{f}`" for f in b["frames"]), ""]
    L += ["## P3 粗看效果（GPU，种子 0–19；只看方向，不作结论）", "",
          "| 条件 | d=0 成功 | d=%.2f 成功 |" % D_MAX, "|---|---|---|"]
    for k in ("10", "100", "te"):
        x, y = r["p3"][f"k{k}_d0"], r["p3"][f"k{k}_dmax"]
        name = "te" if k == "te" else f"k={k}"
        L.append(f"| {name} | {x['n_success']}/{x['n']} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}] | "
                 f"{y['n_success']}/{y['n']} [{y['ci95'][0]:.2f}, {y['ci95'][1]:.2f}] |")
    L += ["", "推的状态（d=%.2f）：" % D_MAX + "；".join(
        f"{k}: {r['p3'][k + '_dmax']['push_status_counts']}，截断 {r['p3'][k + '_dmax']['n_push_clipped']}"
        for k in ("k10", "k100", "kte")), ""]
    (OUT / "checks.md").write_text("\n".join(L), encoding="utf-8")


def analyze() -> None:
    r = dict(p1=p1(), p2=p2(), p3=p3())
    r["frames_cleanup"] = cleanup_frames()
    (OUT / "checks.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(r)
    print((OUT / "checks.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    {"window": window, "analyze": analyze}[sys.argv[1]]()
