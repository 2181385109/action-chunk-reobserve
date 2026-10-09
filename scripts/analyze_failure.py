"""T13 失败案例分析（探索性，只描述，不做检验）。定义和阈值见 results/failure/definitions.md 与 src/chunkre/failure.py。

用法（WSL 内，先 source scripts/env.sh；run_failure.sh 跑完 7 格后自动调用 report）：
  python scripts/analyze_failure.py calib  [--dir results/failure_calib]                 # 标定数据（调试种子 0–9）-> calib.md / calib.json
  python scripts/analyze_failure.py report [--dir results/failure] [--out results/failure] [--label 文字]
输出（--out）：failure_report.md、failure_summary.json、fig_failure_types.png、fig_push_timing.png、
  fig_push_pair_seed<种子>.png、montage_push_seed<种子>.png、fig_delay_jumps.png、fig_delay_joints_seed<种子>.png、
  montage_delay_seed<种子>.png。缺的格子 / 缺的画面在报告里写明并跳过对应部分。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunkre import failure as F  # noqa: E402
from chunkre.metrics import read_episodes  # noqa: E402

CELLS = ["k20__none", "k100__none", "k20__push0.06", "k100__push0.06", "k20__delay10", "k50__delay10", "k100__delay10"]
PUSH_CELLS = ("k20__push0.06", "k100__push0.06")
JUMP_CELLS = ["k20__none", "k100__none", "k20__delay10", "k50__delay10", "k100__delay10"]
CLASS_COLOR = {"success": "#009E73", "never_lifted": "#999999", "lifted_failed": "#D55E00"}
N_PAIRS = 2
LABEL_DEFAULT = "探索性，只描述，不是证据；调试种子 7000–7049，每格 50 回合"


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _f(x, nd=3) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _desc_row(d: dict, nd=3) -> str:
    return f"{d['n']} / {_f(d['median'], nd)} / {_f(d['q25'], nd)}–{_f(d['q75'], nd)} / {_f(d['p90'], nd)} / {_f(d['mean'], nd)}"


def _delay_of(name: str) -> int:
    return int(name.split("delay")[1]) if "delay" in name else 0


def _k_of(name: str) -> str:
    return name.split("__")[0]


# ---------------------------------------------------------------- 读数据
def load_cells(d: Path, names) -> dict:
    out = {}
    for n in names:
        cd = d / n
        if not (cd / "episodes.csv").exists() or not (cd / "traj").is_dir():
            continue
        rows = {int(r["seed"]): r for r in read_episodes(cd / "episodes.csv")}
        eps = F.load_traj_episodes(cd)
        for s, ep in eps.items():
            r = rows[s]
            ep["success"] = int(r["success"])
            ep["n_policy_calls"] = int(r["n_policy_calls"])
            ep["push_step"] = int(r["push_step"]) if r.get("push_step") not in (None, "") else None
            ep["push_status"] = r.get("push_status") or ""
            ep["class"] = F.classify_episode(bool(ep["success"]), ep["cube_pos"][:, 2])
            ep["max_lift"] = F.max_lift(ep["cube_pos"][:, 2])
            fr = cd / "frames" / f"seed{s}"
            ep["frames"] = sorted(int(p.stem[4:]) for p in fr.glob("step*.png")) if fr.is_dir() else []
            ep["frame_dir"] = fr
        cfg = json_load(cd / "config.json")
        out[n] = dict(eps=eps, gl=cfg.get("gl_renderer"), frame_every=(cfg.get("logging") or {}).get("frame_every"))
    return out


def json_load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ---------------------------------------------------------------- 失败类型
def failure_types(cells: dict) -> dict:
    return {n: {c: sum(ep["class"] == c for ep in v["eps"].values()) for c in F.CLASSES} | {"n": len(v["eps"])}
            for n, v in cells.items()}


def fig_failure_types(types: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(types)
    fig, ax = plt.subplots(figsize=(1.3 * len(names) + 2, 4))
    bottom = np.zeros(len(names))
    for c in F.CLASSES:
        v = np.array([types[n][c] for n in names], dtype=float)
        ax.bar(names, v, bottom=bottom, color=CLASS_COLOR[c], label=F.CLASS_LABEL[c], edgecolor="white")
        for i, (b, x) in enumerate(zip(bottom, v)):
            if x > 0:
                ax.text(i, b + x / 2, f"{int(x)}", ha="center", va="center", fontsize=9, color="white")
        bottom += v
    ax.set_ylabel("episodes")
    ax.set_title("Failure types per cell (exploratory)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- Q1：推
def push_episode_stats(ep: dict) -> dict:
    t0, n = ep["push_step"], ep["steps"]
    out = dict(push_step=t0, status=ep["push_status"], first_rp=None, t_close=None, close_dist=None,
               close_before_rp=None, plan_age=None)
    if ep["push_status"] != "pushed" or t0 is None:
        return out
    out["first_rp"] = F.first_repredict_after(ep["repredict"], t0, n)
    tc = F.first_close_after(ep["action"][:, F.GRIP_DIM[F.GRASP_ARM]], t0)
    if tc is not None and "finger_tip" in ep:
        gp = F.grip_point(ep["finger_tip"])
        out.update(t_close=tc, close_dist=F.horiz_dist(gp[tc], ep["cube_pos"][tc]))
        out["close_before_rp"] = bool(out["first_rp"] is None or tc - t0 < out["first_rp"])
        rp = np.flatnonzero(ep["repredict"][:tc + 1])
        out["plan_age"] = int(tc - rp[-1])        # 闭合那一步执行的动作块是多少步之前预测的
    return out


def push_analysis(cells: dict) -> dict:
    res = {}
    for n in PUSH_CELLS:
        if n not in cells:
            continue
        st = {s: push_episode_stats(ep) for s, ep in cells[n]["eps"].items()}
        pushed = {s: v for s, v in st.items() if v["status"] == "pushed"}
        by_cls = {}
        for c in F.CLASSES:
            ss = [s for s in pushed if cells[n]["eps"][s]["class"] == c]
            by_cls[c] = F.describe([pushed[s]["close_dist"] for s in ss])
        res[n] = dict(
            per_seed=st,
            status={k: sum(v["status"] == k for v in st.values()) for k in sorted({v["status"] for v in st.values()})},
            push_step=F.describe([v["push_step"] for v in pushed.values()]),
            first_rp=F.describe([v["first_rp"] for v in pushed.values()]),
            close_delay=F.describe([v["t_close"] - v["push_step"] for v in pushed.values() if v["t_close"] is not None]),
            close_dist=F.describe([v["close_dist"] for v in pushed.values()]),
            close_dist_by_class=by_cls,
            plan_age=F.describe([v["plan_age"] for v in pushed.values()]),
            n_no_close=sum(v["t_close"] is None for v in pushed.values()),
            n_close_before_rp=sum(bool(v["close_before_rp"]) for v in pushed.values() if v["t_close"] is not None),
            n_with_close=sum(v["t_close"] is not None for v in pushed.values()),
        )
    if all(n in cells for n in PUSH_CELLS):
        a, b = (cells[n]["eps"] for n in PUSH_CELLS)
        common = sorted(set(a) & set(b))
        res["pairs"] = {f"{x}{y}": [s for s in common if a[s]["success"] == x and b[s]["success"] == y]
                        for x in (1, 0) for y in (1, 0)}
    return res


def pick_pairs(cells: dict, pairs: list[int]) -> list[int]:
    """k20 成功、k100 失败的配对里，按种子升序先挑两格都有画面的，不够再挑没有画面的。"""
    a, b = (cells[n]["eps"] for n in PUSH_CELLS)
    with_frames = [s for s in pairs if a[s]["frames"] and b[s]["frames"]]
    rest = [s for s in pairs if s not in with_frames]
    return (with_frames + rest)[:N_PAIRS]


def fig_push_timing(pa: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    colors = {"k20__push0.06": "#009E73", "k100__push0.06": "#D55E00"}
    for n in PUSH_CELLS:
        if n not in pa:
            continue
        ps = pa[n]["per_seed"].values()
        rp = [v["first_rp"] for v in ps if v["first_rp"] is not None]
        cd = [100 * v["close_dist"] for v in ps if v["close_dist"] is not None]
        axes[0].hist(rp, bins=np.arange(0, 102, 4), alpha=0.6, color=colors[n], label=_k_of(n))
        axes[1].hist(cd, bins=np.arange(0, max(cd + [10]) + 1, 1), alpha=0.6, color=colors[n], label=_k_of(n))
    axes[0].set_xlabel("steps from push to first re-prediction")
    axes[0].set_ylabel("episodes")
    axes[1].set_xlabel("horizontal grip-cube distance at first close after push (cm)")
    for ax in axes:
        ax.legend(frameon=False)
    fig.suptitle("Push 0.06 m (exploratory)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_push_pair(cells: dict, seed: int, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(11, 7.5), sharex=True)
    for j, n in enumerate(PUSH_CELLS):
        ep = cells[n]["eps"][seed]
        st = push_episode_stats(ep)
        t = np.arange(ep["steps"])
        cube = ep["cube_pos"]
        gp = F.grip_point(ep["finger_tip"]) if "finger_tip" in ep else np.full_like(cube, np.nan)
        for i, (c, lab) in enumerate([(0, "x"), (1, "y")]):
            ax = axes[i, j]
            ax.plot(t, cube[:, c], color="#0072B2", label=f"cube {lab}")
            ax.plot(t, gp[:, c], color="#D55E00", ls="--", label=f"right grip {lab}")
            ax.set_ylabel(f"{lab} (m)")
        dist = np.linalg.norm(gp[:, :2] - cube[:, :2], axis=1)
        ax = axes[2, j]
        ax.plot(t, 100 * dist, color="#000000", label="grip-cube horizontal dist (cm)")
        ax2 = ax.twinx()
        ax2.plot(t, ep["action"][:, F.GRIP_DIM[F.GRASP_ARM]], color="#CC79A7", lw=0.8, label="right gripper cmd")
        ax2.set_ylim(-0.1, 1.3)
        ax.set_ylabel("dist (cm)")
        ax.set_xlabel("step")
        for ax in axes[:, j]:
            if st["push_step"] is not None:
                ax.axvline(st["push_step"], color="red", lw=1.2)
            for r in np.flatnonzero(ep["repredict"]):
                ax.axvline(r, color="#888888", lw=0.5, ls=":")
            if st["t_close"] is not None:
                ax.axvline(st["t_close"], color="#CC79A7", lw=1.2, ls="--")
        res = "success" if ep["success"] else "failure"
        axes[0, j].set_title(f"{_k_of(n)}: {res}; push t={st['push_step']}, first re-predict +{st['first_rp']}, "
                             f"close t={st['t_close']}", fontsize=9)
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[1, 0].legend(fontsize=8, frameon=False)
    h1, l1 = axes[2, 0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[2, 1].legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(f"Seed {seed}: red = push, dotted = re-prediction, pink dashed = first gripper close after push")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _nearest_frame(frames: list[int], t: int, side: str) -> int | None:
    if not frames:
        return None
    if side == "le":
        c = [f for f in frames if f <= t]
        return c[-1] if c else frames[0]
    if side == "ge":
        c = [f for f in frames if f >= t]
        return c[0] if c else frames[-1]
    return min(frames, key=lambda f: abs(f - t))


def montage(tiles: list[list[tuple[Path | None, str]]], path: Path, width: int = 320):
    """tiles：行 × 列，每格 (PNG 路径或 None, 标注)。缩小到 width 宽，标注写在每格上方。"""
    from PIL import Image, ImageDraw

    h = int(width * 3 / 4)
    pad = 18
    rows, cols = len(tiles), max(len(r) for r in tiles)
    canvas = Image.new("RGB", (cols * width, rows * (h + pad)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(tiles):
        for j, (p, text) in enumerate(row):
            x, y = j * width, i * (h + pad)
            if p is not None and Path(p).exists():
                im = Image.open(p).convert("RGB")
                im = im.resize((width, round(im.height * width / im.width)))
                canvas.paste(im.crop((0, 0, width, h)), (x, y + pad))
            draw.text((x + 4, y + 3), text, fill="black")
    canvas.save(path, optimize=True)


def montage_push(cells: dict, seed: int, path: Path) -> bool:
    tiles = []
    for n in PUSH_CELLS:
        ep = cells[n]["eps"][seed]
        if not ep["frames"]:
            return False
        st = push_episode_stats(ep)
        t0 = st["push_step"]
        tc = st["t_close"] if st["t_close"] is not None else ep["steps"] - 1
        picks = [("before push", _nearest_frame(ep["frames"], t0, "le")),
                 ("after push", _nearest_frame(ep["frames"], t0 + 1, "ge")),
                 ("first close" if st["t_close"] is not None else "last frame (no close)",
                  _nearest_frame(ep["frames"], tc, "near"))]
        res = "success" if ep["success"] else "failure"
        tiles.append([(ep["frame_dir"] / f"step{f:04d}.png", f"{_k_of(n)} ({res}) {lab}: step {f}") for lab, f in picks])
    montage(tiles, path)
    return True


# ---------------------------------------------------------------- Q2：延迟
def jump_analysis(cells: dict) -> dict:
    res = {}
    for n in JUMP_CELLS:
        if n not in cells:
            continue
        d = _delay_of(n)
        full, arm, grip, b_full, b_arm, b_grip, back, nsw = [], [], [], [], [], [], [], []
        for ep in cells[n]["eps"].values():
            j = F.switch_jumps(ep["action"], ep["repredict"], ep["steps"], d)
            full += list(j["full"]); arm += list(j["arm"]); grip += list(j["grip"])
            b_full += list(j["base_full"]); b_arm += list(j["base_arm"]); b_grip += list(j["base_grip"])
            back += list(F.jump_backtrack_cos(ep["action"], j["t_exec"]))
            nsw.append(len(j["t_exec"]))
        back = np.asarray(back, dtype=float)
        res[n] = dict(delay=d, jump_full=F.describe(full), jump_arm=F.describe(arm), jump_grip=F.describe(grip),
                      base_full=F.describe(b_full), base_arm=F.describe(b_arm), base_grip=F.describe(b_grip),
                      switches_per_ep=F.describe(nsw), back_cos=F.describe(back),
                      frac_back=float(np.mean(back[~np.isnan(back)] < 0)) if np.any(~np.isnan(back)) else float("nan"),
                      arm_values=arm)
    return res


def fig_delay_jumps(ja: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(ja)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    data = [np.asarray(ja[n]["arm_values"]) for n in names]
    axes[0].boxplot([x[x > 0] if len(x) else [np.nan] for x in data], showfliers=True, whis=(5, 95))
    axes[0].scatter(range(1, len(names) + 1), [ja[n]["base_arm"]["median"] for n in names], marker="_", s=300,
                    color="#0072B2", label="median within-chunk step change")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("12-joint L2 jump at chunk switch (rad)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(range(1, len(names) + 1), [ja[n]["frac_back"] for n in names], color="#D55E00")
    axes[1].set_ylabel("fraction of switches that jump backwards")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=20, ha="right")
    fig.suptitle("Chunk-switch jumps (exploratory; whiskers 5-95%)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pick_delay_failure(cells: dict) -> int | None:
    n = "k20__delay10"
    if n not in cells:
        return None
    eps = cells[n]["eps"]
    cand = [s for s in sorted(eps) if not eps[s]["success"] and eps[s]["frames"]]
    cand = cand or [s for s in sorted(eps) if not eps[s]["success"]]
    return cand[0] if cand else None


JOINT_PLOT = [(7, "right waist"), (8, "right shoulder"), (9, "right elbow"), (13, "right gripper")]


def fig_delay_joints(cells: dict, seed: int, path: Path) -> int | None:
    """返回该回合 12 关节跳变最大的换块执行步（给画面拼图用）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = cells["k20__delay10"]["eps"][seed]
    j = F.switch_jumps(ep["action"], ep["repredict"], ep["steps"], 10)
    t = np.arange(ep["steps"])
    fig, axes = plt.subplots(len(JOINT_PLOT), 1, figsize=(11, 8), sharex=True)
    for ax, (dim, lab) in zip(axes, JOINT_PLOT):
        ax.plot(t, ep["action"][:, dim], color="#D55E00", lw=1, label="executed action")
        if "agent_pos" in ep:
            ax.plot(t, ep["agent_pos"][:, dim], color="#0072B2", lw=1, ls="--", label="observed position")
        for te in j["t_exec"]:
            ax.axvline(te, color="#888888", lw=0.5, ls=":")
        ax.set_ylabel(lab, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[-1].set_xlabel("step (dotted = first executed action of a new chunk = re-prediction step + 10)")
    fig.suptitle(f"k=20, delay 10, seed {seed} ({'success' if ep['success'] else 'failure'}, {ep['class']})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    if len(j["t_exec"]) == 0:
        return None
    return int(j["t_exec"][int(np.argmax(j["arm"]))])


def montage_delay(cells: dict, seed: int, t_sw: int, path: Path) -> bool:
    ep = cells["k20__delay10"]["eps"][seed]
    if not ep["frames"] or t_sw is None:
        return False
    fe = cells["k20__delay10"]["frame_every"] or 1
    want = [t_sw + o * fe for o in range(-3, 5)]
    picks = []
    for w in want:
        f = _nearest_frame(ep["frames"], w, "near")
        if f not in picks:
            picks.append(f)
    row1 = [(ep["frame_dir"] / f"step{f:04d}.png", f"step {f} ({f - t_sw:+d} vs switch)") for f in picks[:4]]
    row2 = [(ep["frame_dir"] / f"step{f:04d}.png", f"step {f} ({f - t_sw:+d} vs switch)") for f in picks[4:8]]
    montage([row1, row2] if row2 else [row1], path)
    return True


# ---------------------------------------------------------------- 报告
def report(d: Path, out: Path, label: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cells = load_cells(d, CELLS)
    if not cells:
        raise SystemExit(f"{d} 下没有可读的格子（需要 episodes.csv + traj/）")
    seeds = sorted({s for v in cells.values() for s in v["eps"]})
    L = [f"# T13 失败案例分析报告", "", f"> **{label}**", "",
         f"数据：`{(d.relative_to(ROOT) if d.is_relative_to(ROOT) else d).as_posix()}`，预训练 ACT，GPU 渲染（`{next(iter(cells.values()))['gl']}`），"
         f"种子 {seeds[0]}–{seeds[-1]}（{len(seeds)} 个）。格子：" + "、".join(f"`{n}`（{len(v['eps'])} 回合）" for n, v in cells.items()) + "。",
         "缺的格子：" + ("、".join(f"`{n}`" for n in CELLS if n not in cells) or "无") + "。",
         "定义和阈值见 `results/failure/definitions.md`（在看 7000–7049 数据之前提交，之后不改）。",
         "本报告由 `scripts/analyze_failure.py report` 自动生成，数字全部由各格 `episodes.csv` 和 `traj/*.npz` 算出。", ""]
    summary = dict(label=label, seeds=[seeds[0], seeds[-1], len(seeds)], cells={n: len(v["eps"]) for n, v in cells.items()})

    # 1 失败类型
    types = failure_types(cells)
    summary["failure_types"] = types
    fig_failure_types(types, out / "fig_failure_types.png")
    L += ["## 1. 失败类型（每格三类各多少）", "",
          "| 格子 | 回合 | 成功 | 从没抓起 | 抓起但没完成 |", "|---|---|---|---|---|"]
    L += [f"| {n} | {t['n']} | {t['success']} | {t['never_lifted']} | {t['lifted_failed']} |" for n, t in types.items()]
    L += ["", "![失败类型](fig_failure_types.png)", ""]

    # 2 推
    pa = push_analysis(cells)
    L += ["## 2. Q1 推 0.06 m：为什么 k=100 失败多、k=20 成功多", ""]
    if not any(n in pa for n in PUSH_CELLS):
        L += ["（没有推的格子，跳过）", ""]
    else:
        L += ["推的时刻在每回合由种子决定，同一种子在 k=20 / k=100 下相同。“推后第一次重新预测”= 推之后（t > 推的步）第一次重新预测距推的步数；"
              "“推后第一次闭合”= 推之后右夹爪指令第一次完成“张开 → 闭合”（定义见 definitions.md §3），距离 = 那一步右手两指尖中点与方块中心的水平距离；“计划年龄”= 闭合那一步执行的动作块是多少步之前预测的（k=20 最多 19，k=100 最多 99）。", "",
              "| 格子 | 推了 / 状态 | 推的步 中位数（四分位） | 推后第一次重新预测（步）：n / 中位数 / 四分位 / p90 / 均值 | 推后第一次闭合距推（步）中位数 | 闭合时水平距离（m）：n / 中位数 / 四分位 / p90 / 均值 | 闭合时计划年龄（步）中位数（范围） | 推后没有闭合 | 闭合在推后第一次重新预测之前 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for n in PUSH_CELLS:
            if n not in pa:
                continue
            r = pa[n]
            stt = ", ".join(f"{k} {v}" for k, v in r["status"].items())
            L.append(f"| {n} | {stt} | {_f(r['push_step']['median'], 0)}（{_f(r['push_step']['q25'], 0)}–{_f(r['push_step']['q75'], 0)}） | "
                     f"{_desc_row(r['first_rp'], 1)} | {_f(r['close_delay']['median'], 0)} | {_desc_row(r['close_dist'])} | "
                     f"{_f(r['plan_age']['median'], 0)}（{_f(r['plan_age']['min'], 0)}–{_f(r['plan_age']['max'], 0)}） | "
                     f"{r['n_no_close']} | {r['n_close_before_rp']}/{r['n_with_close']} |")
        L += ["", "按结局分开的闭合时水平距离（m，中位数 / n）：", "",
              "| 格子 | 成功 | 从没抓起 | 抓起但没完成 |", "|---|---|---|---|"]
        for n in PUSH_CELLS:
            if n in pa:
                b = pa[n]["close_dist_by_class"]
                L.append(f"| {n} | " + " | ".join(f"{_f(b[c]['median'])}（n={b[c]['n']}）" for c in F.CLASSES) + " |")
        fig_push_timing(pa, out / "fig_push_timing.png")
        L += ["", "![推后时间](fig_push_timing.png)", ""]
        summary["push"] = {n: {k: v for k, v in pa[n].items() if k != "per_seed"} for n in PUSH_CELLS if n in pa}
        if "pairs" in pa:
            p = pa["pairs"]
            summary["push"]["pairs"] = {k: len(v) for k, v in p.items()}
            L += ["同一种子配对（k=20 结局 / k=100 结局）：" + "，".join(
                f"{'成' if k[0] == '1' else '败'}/{'成' if k[1] == '1' else '败'} {len(v)}" for k, v in p.items()) + "。", ""]
            chosen = pick_pairs(cells, p["10"])
            summary["push"]["chosen_pairs"] = chosen
            if not chosen:
                L += ["没有 k=20 成功、k=100 失败的配对，不画配对图。", ""]
            L += [f"挑选规则：k=20 成功、k=100 失败的配对里，按种子升序先取两格都存了画面的，不够再取没画面的（只画曲线）。挑中：{chosen}。", ""]
            for s in chosen:
                fig_push_pair(cells, s, out / f"fig_push_pair_seed{s}.png")
                L += [f"### 种子 {s}", "", f"![配对 {s}](fig_push_pair_seed{s}.png)", ""]
                st20, st100 = (push_episode_stats(cells[n]["eps"][s]) for n in PUSH_CELLS)
                L += [f"- k=20：推在第 {st20['push_step']} 步，推后 +{st20['first_rp']} 步重新预测，推后第一次闭合在第 {st20['t_close']} 步，水平距离 {_f(st20['close_dist'])} m；",
                      f"- k=100：推在第 {st100['push_step']} 步，推后 +{st100['first_rp']} 步重新预测，推后第一次闭合在第 {st100['t_close']} 步，水平距离 {_f(st100['close_dist'])} m。", ""]
                if montage_push(cells, s, out / f"montage_push_seed{s}.png"):
                    L += [f"![画面 {s}](montage_push_seed{s}.png)", "（上行 k=20，下行 k=100；推之前 / 推之后 / 推后第一次闭合，最近的存图步）", ""]
                else:
                    L += ["（这个种子没有存画面，不做拼图）", ""]
        else:
            L += ["（k=20 / k=100 推的格子不全，不做配对）", ""]

    # 3 延迟
    ja = jump_analysis(cells)
    L += ["## 3. Q2 延迟 10 步：换块时刻的跳变", ""]
    if not ja:
        L += ["（没有相关格子，跳过）", ""]
    else:
        L += ["跳变 = 新块第一个**被执行**的动作（延迟 d 时 = 第 t_r + d 步，t_r 为重新预测的步，不含第 0 步的初始块）减去上一步执行的动作；"
              "“块内”= 其余相邻两步（参照）。单位：关节为弧度，夹爪为归一化开合度。“往回跳比例”= 跳变（12 关节）与跳变前 5 步运动方向的余弦 < 0 的比例。"
              "k=50 无扰动不在 7 格内，所以没有。", "",
              "| 格子 | 12 关节 L2 跳变：换块数 n / 中位数 / 四分位 / p90 / 均值 | 块内 12 关节 中位数 | 换块 2 夹爪 中位数 / p90 | 块内 2 夹爪 中位数 | 换块 14 维 中位数 / p90 | 往回跳比例 | 每回合换块次数 中位数（最小–最大） |",
              "|---|---|---|---|---|---|---|---|"]
        for n, r in ja.items():
            L.append(f"| {n} | {_desc_row(r['jump_arm'], 4)} | {_f(r['base_arm']['median'], 4)} | "
                     f"{_f(r['jump_grip']['median'], 4)} / {_f(r['jump_grip']['p90'], 4)} | {_f(r['base_grip']['median'], 4)} | "
                     f"{_f(r['jump_full']['median'], 4)} / {_f(r['jump_full']['p90'], 4)} | {_f(r['frac_back'], 2)} | "
                     f"{_f(r['switches_per_ep']['median'], 0)}（{_f(r['switches_per_ep']['min'], 0)}–{_f(r['switches_per_ep']['max'], 0)}） |")
        fig_delay_jumps(ja, out / "fig_delay_jumps.png")
        L += ["", "![换块跳变](fig_delay_jumps.png)", ""]
        summary["delay"] = {n: {k: v for k, v in r.items() if k != "arm_values"} for n, r in ja.items()}
        s = pick_delay_failure(cells)
        summary["delay_example_seed"] = s
        if s is None:
            L += ["（没有 k=20 延迟 10 的失败回合，不画关节轨迹）", ""]
        else:
            t_sw = fig_delay_joints(cells, s, out / f"fig_delay_joints_seed{s}.png")
            L += [f"### k=20 延迟 10 失败回合：种子 {s}（挑选规则：失败且存了画面的种子里最小的）", "",
                  f"![关节轨迹 {s}](fig_delay_joints_seed{s}.png)", ""]
            if montage_delay(cells, s, t_sw, out / f"montage_delay_seed{s}.png"):
                L += [f"画面：这一回合 12 关节跳变最大的换块执行步 {t_sw} 前后。", "",
                      f"![画面 {s}](montage_delay_seed{s}.png)", ""]
            else:
                L += ["（这个种子没有存画面，不做拼图）", ""]
    L += ["## 注意", "",
          "- 全部是**探索性描述**：每格 50 回合、调试种子，不做假设检验，不能当证据；机制解释只是与数据相符的说法。",
          "- GPU 渲染逐回合不完全可复现（T2），同一种子重跑结局可能不同；本报告与正式结果（种子 200000–200199）不是同一批回合。",
          "- 指尖位置 = 手指网格最前端 5 mm 内顶点的平均（definitions.md §3），不是接触点。"]
    (out / "failure_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "failure_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
                                              encoding="utf-8")
    print(f"写出 {out / 'failure_report.md'}")
    return summary


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------- 标定
def calib(d: Path) -> dict:
    names = sorted(p.name for p in d.iterdir() if (p / "traj").is_dir())
    cells = load_cells(d, names)
    L = ["# T13 失败分类阈值标定（调试种子 0–9，GPU，每格 10 回合）", "",
         "由 `scripts/analyze_failure.py calib` 生成。用来定 definitions.md 里的阈值；**不是结果**。", "",
         "| 格子 | 种子 | 成功 | 步数 | 初始高度 (m) | 最大抬升 (m) | 右夹爪闭合步（全部） | 闭合时右手指尖中点-方块水平距离 (m) | 左夹爪闭合步 | 推的步 / 状态 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for n, v in cells.items():
        for s, ep in sorted(v["eps"].items()):
            z = ep["cube_pos"][:, 2]
            cr = F.close_events(ep["action"][:, 13])
            cl = F.close_events(ep["action"][:, 6])
            gp = F.grip_point(ep["finger_tip"])
            dist = [F.horiz_dist(gp[t], ep["cube_pos"][t]) for t in cr]
            rows.append(dict(cell=n, seed=s, success=ep["success"], steps=ep["steps"], rest=F.rest_height(z),
                             lift=ep["max_lift"], close_r=cr, dist_r=dist, close_l=cl, push=ep["push_step"],
                             status=ep["push_status"]))
            L.append(f"| {n} | {s} | {ep['success']} | {ep['steps']} | {F.rest_height(z):.4f} | {ep['max_lift']:.3f} | {cr} | "
                     f"{[round(x, 3) for x in dist]} | {cl} | {ep['push_step'] or ''} {ep['push_status']} |")
    succ = [r for r in rows if r["success"]]
    fail = [r for r in rows if not r["success"]]
    first_dist_succ = [r["dist_r"][0] for r in succ if r["dist_r"]]
    agg = dict(n=len(rows), n_success=len(succ),
               rest_min=min(r["rest"] for r in rows), rest_max=max(r["rest"] for r in rows),
               lift_success_min=min(r["lift"] for r in succ) if succ else None,
               lift_fail=sorted(round(r["lift"], 4) for r in fail),
               success_all_right_close_first=all(r["close_r"] and (not r["close_l"] or r["close_r"][0] < r["close_l"][0])
                                                 for r in succ),
               first_close_dist_success=F.describe(first_dist_succ))
    L += ["", "## 汇总", "", "```", json.dumps(agg, indent=2, ensure_ascii=False, default=_json_default), "```"]
    (d / "calib.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (d / "calib.json").write_text(json.dumps(dict(summary=agg, episodes=rows), indent=2, ensure_ascii=False,
                                             default=_json_default), encoding="utf-8")
    print("\n".join(L[-len(json.dumps(agg, indent=2).splitlines()) - 2:]))
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["calib", "report"])
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--label", default=LABEL_DEFAULT)
    a = ap.parse_args(argv)
    if a.cmd == "calib":
        calib(_abs(a.dir or "results/failure_calib"))
    else:
        d = _abs(a.dir or "results/failure")
        report(d, _abs(a.out) if a.out else d, a.label)


if __name__ == "__main__":
    main()
