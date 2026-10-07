"""T5 观测噪声 / 动作延迟检查（结果 results/t5_checks/；一键运行 scripts/t5_checks.sh）。

子命令：
  n2       噪声确实生效（进程内直接跑 run_batch，GPU）：aloha 与 PushT 各 10 个回合（种子 0–9）、前 MAX_STEPS 步，
           σ = 建议最大档；在若干步记录“加噪图像 − 原图”，算经验标准差（应接近 σ，截断会让它略小）；
           同时统计原图本身的像素标准差（用来定 σ 档位）；导出加噪前后对比 PNG（原图 | 三档 σ，最大档是回路里实际加的）。
  analyze  读 N1/N3/N4 的运行结果，做比较，写 checks.json / checks.md。

N1 强度 0 = 无扰动：aloha CPU 渲染，种子 0–9，k=100，--log-traj：n1_act_none vs n1_act_noise0 / n1_act_delay0；
   DP（PushT 不经 OpenGL），种子 0–9，B=10：n1_dp_none vs n1_dp_noise0 / n1_dp_delay0。
   逐回合结果（除墙钟列和扰动列外所有列）完全一致，且整回合实际动作（轨迹 action）逐位相同。
N3 延迟确实生效：aloha N4 的三个延迟运行 + DP 的 n3_dp_delay（--log-traj 里有 policy_action）：
   第 t 步执行的动作 = 第 t−d 步策略输出（逐位），t < d 时 = 第 0 步观测的 agent_pos（逐位）。
N4 粗看效果：aloha GPU，种子 0–19，k ∈ {10, 100} + te，σ = SIGMA_MAX_ACT、d = D_MAX_ACT，只报成功率。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunkre.metrics import TIME_COLUMNS, read_episodes, summarize_csv  # noqa: E402
from chunkre.perturb import PERTURB_COLUMNS, ObsNoise, camera_arrays  # noqa: E402

OUT = ROOT / "results" / "t5_checks"
# 建议档位（依据见 HANDOFF.md；最大档用于 N2 / N4）
SIGMA_LEVELS = {"act": (4.0, 8.0, 16.0), "dp": (4.0, 8.0, 16.0)}
D_LEVELS = {"act": (2, 5, 10), "dp": (1, 2, 4)}
SIGMA_MAX_ACT, D_MAX_ACT = SIGMA_LEVELS["act"][-1], D_LEVELS["act"][-1]
D_MAX_DP = D_LEVELS["dp"][-1]
N2_STEPS = (0, 10, 20, 30, 40, 59)
N2_MAX_STEPS = 60
N4_RUNS = [(p, k) for p in ("noise", "delay") for k in ("10", "100", "te")]


# ----------------------------------------------------------------------------------------------
# N2
# ----------------------------------------------------------------------------------------------
class CaptureNoise(ObsNoise):
    """ObsNoise + 在指定步把（原图，加噪图）复制一份留作统计；不改变加噪本身。"""

    def __init__(self, sigma, steps):
        super().__init__(sigma)
        self.cap_steps = set(steps)
        self.caps = []

    def on_obs(self, t, obs, alive):
        new = super().on_obs(t, obs, alive)
        if t in self.cap_steps:
            for (name, img), (_, nimg) in zip(camera_arrays(obs), camera_arrays(new)):
                for i in np.flatnonzero(alive):
                    self.caps.append(dict(t=int(t), i=int(i), seed=self.seeds[i], cam=name,
                                          orig=np.array(img[i]), noisy=np.array(nimg[i])))
        return new


def _n2_model(model: str) -> dict:
    from chunkre.config import EvalConfig
    from chunkre.runner import CallCounter, build_policy, gl_renderer_of, make_vec_env, run_batch, set_determinism

    sigma = SIGMA_LEVELS[model][-1]
    cfg = EvalConfig(model=model, seeds="0-9", batch_size=10, max_steps=N2_MAX_STEPS)
    set_determinism(True)
    policy, _ = build_policy(cfg)
    counter = CallCounter(policy)
    env = make_vec_env(model, 10)
    cap = CaptureNoise(sigma, N2_STEPS)
    try:
        run_batch(env, policy, counter, cfg.seeds, cfg.device, cfg.max_steps, perturb=cap)
        renderer = gl_renderer_of(env)
    finally:
        env.close()
        counter.remove()
    diffs = np.concatenate([(c["noisy"].astype(np.float64) - c["orig"]).ravel() for c in cap.caps])
    per_cap = [float((c["noisy"].astype(np.float64) - c["orig"]).std()) for c in cap.caps]
    img_std = [float(c["orig"].astype(np.float64).std()) for c in cap.caps]
    # 像素级“局部”标准差：每张图按 8×8 小块算标准差再取中位数（看平坦区域里噪声有多显眼）
    loc = []
    for c in cap.caps:
        g = c["orig"].astype(np.float64).mean(axis=2)
        h, w = (g.shape[0] // 8) * 8, (g.shape[1] // 8) * 8
        blk = g[:h, :w].reshape(h // 8, 8, w // 8, 8).std(axis=(1, 3))
        loc.append(float(np.median(blk)))
    sat = float(np.mean([np.mean((c["orig"] == 0) | (c["orig"] == 255)) for c in cap.caps]))
    # 离线：同一批原图上套三档 σ（新的 ObsNoise，种子同回合），算经验标准差
    levels = {}
    for s in SIGMA_LEVELS[model]:
        d = []
        for c in cap.caps:
            p = ObsNoise(s)
            p.begin_batch(None, [c["seed"]])
            d.append((p.on_obs(0, {"pixels": c["orig"][None]}, np.ones(1, bool))["pixels"][0].astype(np.float64)
                      - c["orig"]).ravel())
        d = np.concatenate(d)
        levels[str(s)] = dict(emp_std=float(d.std()), expected_no_clip=float(np.sqrt(s * s + 1 / 12)))
    _n2_png(model, cap)
    return dict(
        model=model, sigma_in_loop=sigma, gl_renderer=renderer, steps=sorted(cap.cap_steps),
        n_images=len(cap.caps), image_shape=list(cap.caps[0]["orig"].shape),
        emp_std_noisy_minus_orig=float(diffs.std()), emp_mean_noisy_minus_orig=float(diffs.mean()),
        expected_std_if_no_clip=float(np.sqrt(sigma * sigma + 1 / 12)),
        per_image_emp_std_min_max=[min(per_cap), max(per_cap)],
        image_pixel_std_mean=float(np.mean(img_std)), image_pixel_std_min_max=[min(img_std), max(img_std)],
        image_local8x8_std_median=float(np.median(loc)),
        frac_pixels_saturated_orig=sat,
        levels_offline=levels,
        note="经验标准差 = 所有记录图像所有像素通道上 (加噪 − 原图) 的标准差；rint 取整使无截断时期望为 sqrt(σ²+1/12)",
    )


def _n2_png(model: str, cap: CaptureNoise) -> None:
    from PIL import Image

    c = next(x for x in cap.caps if x["seed"] == 0 and x["t"] == 20)
    tiles = [c["orig"]]
    for s in SIGMA_LEVELS[model][:-1]:
        p = ObsNoise(s)
        p.begin_batch(None, [0])
        tiles.append(p.on_obs(0, {"pixels": c["orig"][None]}, np.ones(1, bool))["pixels"][0])
    tiles.append(c["noisy"])                 # 最大档：回路里策略实际看到的图
    im = np.concatenate(tiles, axis=1)
    img = Image.fromarray(im)
    if model == "dp":
        img = img.resize((img.width * 4, img.height * 4), Image.NEAREST)
    else:
        img = img.resize((img.width // 2, img.height // 2), Image.NEAREST)
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / f"n2_{model}_seed0_step20.png")
    # 局部放大（aloha 原图左上角起 120×160 区域的原图 vs 最大档，2 倍最近邻），看清噪声颗粒
    if model == "act":
        h0, w0 = 180, 240
        crop = np.concatenate([c["orig"][h0:h0 + 120, w0:w0 + 160], c["noisy"][h0:h0 + 120, w0:w0 + 160]], axis=1)
        Image.fromarray(crop).resize((crop.shape[1] * 3, crop.shape[0] * 3), Image.NEAREST).save(
            OUT / "n2_act_seed0_step20_zoom.png")


def n2(models) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "n2.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for m in models:
        res[m] = _n2_model(m)
        print(json.dumps({k: v for k, v in res[m].items() if k != "levels_offline"}, indent=1, ensure_ascii=False))
        print(json.dumps(res[m]["levels_offline"], ensure_ascii=False))
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# analyze
# ----------------------------------------------------------------------------------------------
def _load_traj(run: Path) -> list:
    return [dict(np.load(p)) for p in sorted((run / "traj").glob("batch_*.npz"))]


def _compare(a: Path, b: Path) -> dict:
    ra, rb = read_episodes(a / "episodes.csv"), read_episodes(b / "episodes.csv")
    skip = set(TIME_COLUMNS) | set(PERTURB_COLUMNS)
    cols = [c for c in ra[0] if c not in skip]
    key = lambda r: int(r["seed"])  # noqa: E731
    ra, rb = sorted(ra, key=key), sorted(rb, key=key)
    assert [key(r) for r in ra] == [key(r) for r in rb]
    same_ep = [all(x[c] == y[c] for c in cols) for x, y in zip(ra, rb)]
    ta, tb = _load_traj(a), _load_traj(b)
    acts_same, keys_same = [], {}
    for za, zb in zip(ta, tb):
        assert np.array_equal(za["seeds"], zb["seeds"]) and np.array_equal(za["steps"], zb["steps"])
        for i in range(len(za["seeds"])):
            n = int(za["steps"][i])
            acts_same.append(bool(np.array_equal(za["action"][:n, i], zb["action"][:n, i])))
        for k in za:
            keys_same[k] = keys_same.get(k, True) and k in zb and za[k].shape == zb[k].shape and \
                bool(np.array_equal(za[k], zb[k], equal_nan=za[k].dtype.kind == "f"))
    return dict(a=str(a.relative_to(ROOT)), b=str(b.relative_to(ROOT)), n=len(ra), compared_columns=cols,
                episodes_identical=int(sum(same_ep)), full_episode_actions_identical=int(sum(acts_same)),
                traj_arrays_identical={k: bool(v) for k, v in keys_same.items()},
                success_a=int(sum(int(r["success"]) for r in ra)), success_b=int(sum(int(r["success"]) for r in rb)))


def n1() -> dict:
    out = {}
    for m in ("act", "dp"):
        for p in ("noise0", "delay0"):
            out[f"{m}_{p}"] = _compare(OUT / f"n1_{m}_none", OUT / f"n1_{m}_{p}")
    return out


def _check_delay(run: Path, d: int) -> dict:
    n_steps = n_mismatch = n_hold = n_hold_bad = n_ep = 0
    for z in _load_traj(run):
        act, pol, ap = z["action"], z["policy_action"], z["agent_pos"]
        for i in range(len(z["seeds"])):
            n_ep += 1
            n = int(z["steps"][i])
            for t in range(n):
                n_steps += 1
                if t < d:
                    n_hold += 1
                    n_hold_bad += int(not np.array_equal(act[t, i], ap[0, i]))
                else:
                    n_mismatch += int(not np.array_equal(act[t, i], pol[t - d, i]))
    # 反例：如果拿错了延迟（d±1），应当大量不一致（说明检查本身有区分力）
    z = _load_traj(run)[0]
    n0 = int(z["steps"][0])
    off = {str(dd): int(sum(not np.array_equal(z["action"][t, 0], z["policy_action"][t - dd, 0]) for t in range(d + 1, n0)))
           for dd in (d - 1, d + 1) if dd >= 0}
    return dict(run=str(run.relative_to(ROOT)), d=d, n_episodes=n_ep, n_steps=n_steps,
                n_delayed_steps=n_steps - n_hold, n_delayed_mismatch=n_mismatch,
                n_hold_steps=n_hold, n_hold_mismatch=n_hold_bad,
                wrong_d_mismatch_seed0=off)


def n3() -> dict:
    out = {f"act_{k}": _check_delay(OUT / f"n4_delay_{k}", D_MAX_ACT) for k in ("10", "100", "te")}
    out["dp_k8"] = _check_delay(OUT / "n3_dp_delay", D_MAX_DP)
    return out


def n4() -> dict:
    out = {}
    for p, k in N4_RUNS:
        s = summarize_csv(OUT / f"n4_{p}_{k}" / "episodes.csv")
        out[f"{p}_{k}"] = dict(n=s["n_episodes"], success=s["n_success"], ci95=s["success_ci95_wilson"],
                               mean_policy_calls=s["mean_policy_calls"], mean_smoothness=s["mean_smoothness"],
                               wall_s_per_episode=s["wall_s_per_episode"])
    return out


def analyze() -> None:
    n2res = json.loads((OUT / "n2.json").read_text(encoding="utf-8")) if (OUT / "n2.json").exists() else {}
    res = dict(n1=n1(), n2=n2res, n3=n3(), n4=n4(),
               levels=dict(sigma=SIGMA_LEVELS, d=D_LEVELS))
    (OUT / "checks.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    L = ["# T5 检查结果（由 scripts/t5_checks.py analyze 生成）", ""]
    L.append("## N1 强度 0 = 无扰动（逐回合 / 整回合动作）")
    L.append("| 对比 | 逐回合一致 | 整回合动作逐位相同 | 轨迹所有数组相同 | 成功 none / 强度0 |")
    L.append("|---|---|---|---|---|")
    for k, v in res["n1"].items():
        L.append(f"| {v['a']} vs {v['b'].split('/')[-1]} | {v['episodes_identical']}/{v['n']} | "
                 f"{v['full_episode_actions_identical']}/{v['n']} | {all(v['traj_arrays_identical'].values())} | "
                 f"{v['success_a']} / {v['success_b']} |")
    L += ["", "## N2 噪声确实生效（前 60 步里第 " + ", ".join(map(str, N2_STEPS)) + " 步，种子 0–9）"]
    L.append("| 模型 | σ | 实测 std(加噪−原图) | 无截断期望 sqrt(σ²+1/12) | 原图像素 std 平均 | 原图 8×8 局部 std 中位数 | 原图饱和像素比例 |")
    L.append("|---|---|---|---|---|---|---|")
    for m, v in n2res.items():
        L.append(f"| {m} | {v['sigma_in_loop']:g} | {v['emp_std_noisy_minus_orig']:.3f} | {v['expected_std_if_no_clip']:.3f} | "
                 f"{v['image_pixel_std_mean']:.1f} | {v['image_local8x8_std_median']:.2f} | {v['frac_pixels_saturated_orig']:.3f} |")
    for m, v in n2res.items():
        L.append(f"- {m} 三档离线实测：" + "；".join(f"σ={s}：{x['emp_std']:.3f}（期望 {x['expected_no_clip']:.3f}）"
                                             for s, x in v["levels_offline"].items()))
    L.append("- 对比图：`n2_act_seed0_step20.png`、`n2_dp_seed0_step20.png`（原图 | 三档 σ，最右为回路里实际加的最大档）、"
             "`n2_act_seed0_step20_zoom.png`（局部放大：原图 | 最大档）")
    L += ["", "## N3 延迟确实生效（执行动作 = 第 t−d 步策略输出；t<d 为第 0 步 agent_pos）"]
    L.append("| 运行 | d | 回合 | 延迟步 / 不一致 | 保持步 / 不一致 | 错用 d±1 时不一致步数（种子 0） |")
    L.append("|---|---|---|---|---|---|")
    for k, v in res["n3"].items():
        L.append(f"| {v['run']} | {v['d']} | {v['n_episodes']} | {v['n_delayed_steps']} / {v['n_delayed_mismatch']} | "
                 f"{v['n_hold_steps']} / {v['n_hold_mismatch']} | {v['wrong_d_mismatch_seed0']} |")
    L += ["", f"## N4 粗看效果（aloha GPU，种子 0–19；σ={SIGMA_MAX_ACT:g}，d={D_MAX_ACT}；只看方向）"]
    L.append("| 条件 | 无扰动（T4 P3 d=0） | 噪声 σ 最大档 | 延迟 d 最大档 |")
    L.append("|---|---|---|---|")
    ref = {}
    for k in ("10", "100", "te"):
        p = ROOT / "results" / "t4_checks" / f"p3_k{k}_d0" / "episodes.csv"
        if p.exists():
            s = summarize_csv(p)
            ref[k] = f"{s['n_success']}/{s['n_episodes']}"
    for k in ("10", "100", "te"):
        a, b = res["n4"][f"noise_{k}"], res["n4"][f"delay_{k}"]
        L.append(f"| k={k} | {ref.get(k, '—')} | {a['success']}/{a['n']} | {b['success']}/{b['n']} |")
    (OUT / "checks.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["n2", "analyze"])
    ap.add_argument("--models", default="act,dp")
    a = ap.parse_args()
    if a.cmd == "n2":
        n2(a.models.split(","))
    else:
        analyze()
