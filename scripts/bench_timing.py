"""V5 耗时基准：不同环境数 B 下，“一次策略预测”和“一次环境步进+渲染”各要多久，并估算每回合墙钟。

用法（WSL 内；必须单独跑，不要和别的 GPU/渲染任务并行）：
  python scripts/bench_timing.py measure act --Bs 1,10,25,50
  python scripts/bench_timing.py measure dp  --Bs 1,10,25,50
  python scripts/bench_timing.py estimate          # 读 timing_raw_*.csv，写 results/t2_checks/timing.csv
每个 B 测完立刻追加写入 timing_raw_<model>.csv，所以 B 大到内存不够被杀时，前面的结果还在。

测法（真实环境 + 真实策略，种子 0..B-1，只在调试区间）：
- 策略设成 k=1（每步都重新预测），先跑 WARMUP 步预热，再测 N 步：
  predict_s = 每次 select_action（含一次动作块预测）的时间；env_step_s = 每次 env.step（B 个环境步进+渲染）；
  prep_s = 观测预处理+拷到 GPU 的时间。
- 再把 n_action_steps 改成上限（ACT 100 / DP 15）并 reset，测“只从队列取动作、不预测”的 select_action 时间 pop_s。
- B=10 时另测确定性开关关闭时的 predict_s（看确定性开关有没有拖慢）。
估算（每批跑满 T 步：B>=10 时一批里几乎总有失败回合跑满 T，B 小时是偏保守的上界）：
  一批墙钟 = T*(env_step_s + prep_s) + ceil(T/k)*predict_s + (T - ceil(T/k))*pop_s
  每回合墙钟 = 一批墙钟 / B          （ACT T=400，DP T=300；te 与 k=1 相同的调用次数）
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "t2_checks"
T = {"act": 400, "dp": 300}
KS = {"act": [1, 2, 5, 10, 25, 50, 100, "te"], "dp": [1, 2, 4, 8, 15]}
N_MEAS = {"act": 10, "dp": 3}
WARMUP = 2
RAW_COLS = ["model", "B", "predict_s", "predict_s_nondet", "pop_s", "env_step_s", "prep_s",
            "n_predict_meas", "make_env_s", "peak_rss_GB", "gpu_peak_GB"]


def measure(model: str, Bs: list[int], variant: str = "") -> None:
    import numpy as np
    import torch
    from lerobot.envs.utils import add_envs_task, preprocess_observation

    from chunkre.config import load_config
    from chunkre.runner import build_policy, make_vec_env, set_determinism, set_global_seed
    from chunkre.seed_guard import check_seeds

    raw = OUT / f"timing_raw_{model}{variant}.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    new = not raw.exists()
    cfg = load_config(None, dict(model=model, seeds="0-0", batch_size=1, k=1))
    set_determinism(True)
    policy, _ = build_policy(cfg)
    ub = 100 if model == "act" else 15

    def sync():
        torch.cuda.synchronize()

    for B in Bs:
        seeds = list(range(B))
        check_seeds(seeds)
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        env = make_vec_env(model, B)
        make_env_s = time.perf_counter() - t
        set_global_seed(0)
        obs, _ = env.reset(seed=seeds)

        def step_once(times):
            nonlocal obs
            sync(); a = time.perf_counter()
            o = preprocess_observation(obs)
            o = {k: v.to("cuda", non_blocking=True) for k, v in o.items()}
            o = add_envs_task(env, o)
            sync(); b = time.perf_counter()
            with torch.inference_mode():
                act = policy.select_action(o)
            act = act.to("cpu").numpy()
            c = time.perf_counter()
            obs, *_ = env.step(act)
            d = time.perf_counter()
            times.append((b - a, c - b, d - c))

        # k=1：每步预测
        policy.config.n_action_steps = 1
        policy.reset()
        tk1 = []
        for i in range(WARMUP + N_MEAS[model]):
            step_once(tk1)
        tk1 = tk1[WARMUP:]
        # 确定性开关关闭时的预测耗时（只在 B=10 测）
        nondet = float("nan")
        if B == 10:
            set_determinism(False)
            tn = []
            for i in range(1 + N_MEAS[model]):
                step_once(tn)
            nondet = float(np.mean([x[1] for x in tn[1:]]))
            set_determinism(True)
        # 只出队、不预测
        policy.config.n_action_steps = ub
        policy.reset()
        tp = []
        for i in range(1 + 8):
            step_once(tp)
        tp = tp[1:]
        row = dict(
            model=model + variant, B=B,
            predict_s=float(np.mean([x[1] for x in tk1])), predict_s_nondet=nondet,
            pop_s=float(np.mean([x[1] for x in tp])),
            env_step_s=float(np.mean([x[2] for x in tk1 + tp])),
            prep_s=float(np.mean([x[0] for x in tk1 + tp])),
            n_predict_meas=len(tk1), make_env_s=make_env_s,
            peak_rss_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
            gpu_peak_GB=torch.cuda.max_memory_allocated() / 1024 ** 3,
        )
        env.close()
        del env
        gc.collect()
        with open(raw, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RAW_COLS)
            if new:
                w.writeheader()
                new = False
            w.writerow(row)
        print(f"[bench {model}{variant}] B={B}: 预测 {row['predict_s'] * 1000:.1f} ms，出队 {row['pop_s'] * 1000:.2f} ms，"
              f"环境步进 {row['env_step_s'] * 1000:.1f} ms，预处理 {row['prep_s'] * 1000:.1f} ms，"
              f"峰值内存 {row['peak_rss_GB']:.2f} GB，GPU 峰值 {row['gpu_peak_GB']:.2f} GB", flush=True)


def est_episode_s(model: str, k, r: dict) -> float:
    Tm = T[model]
    n_pred = Tm if k == "te" else math.ceil(Tm / int(k))
    batch = Tm * (r["env_step_s"] + r["prep_s"]) + n_pred * r["predict_s"] + (Tm - n_pred) * r["pop_s"]
    return batch / r["B"]


def estimate() -> None:
    rows = []
    for name in ["act", "act_cpu", "dp"]:
        model = name.split("_")[0]
        p = OUT / f"timing_raw_{name}.csv"
        if not p.exists():
            continue
        with open(p, newline="", encoding="utf-8") as f:
            raw = [{k: (float(v) if k not in ("model",) else v) for k, v in r.items()} for r in csv.DictReader(f)]
        for r in raw:
            r["B"] = int(r["B"])
            for k in KS[model]:
                rows.append(dict(model=name, k=k, B=r["B"], est_episode_s=est_episode_s(model, k, r),
                                 predict_s=r["predict_s"], env_step_s=r["env_step_s"], prep_s=r["prep_s"],
                                 pop_s=r["pop_s"], T=T[model]))
    with open(OUT / "timing.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # markdown 表：行 = k，列 = B（也写入 timing_table.md）
    md = []
    for name in ["act", "act_cpu", "dp"]:
        model = name.split("_")[0]
        rs = [r for r in rows if r["model"] == name]
        if not rs:
            continue
        Bs = sorted({r["B"] for r in rs})
        title = {"act": "ACT（GPU 渲染）", "act_cpu": "ACT（CPU 渲染 llvmpipe，对照）", "dp": "DP"}[name]
        md.append(f"\n{title}：每回合估算墙钟（秒；T={T[model]}，每批按跑满 T 步计）\n")
        md.append("| k | " + " | ".join(f"B={b}" for b in Bs) + " |")
        md.append("|---|" + "---|" * len(Bs))
        for k in KS[model]:
            vals = [next(r["est_episode_s"] for r in rs if r["k"] == k and r["B"] == b) for b in Bs]
            md.append(f"| {k} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    (OUT / "timing_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def eta(model: str, k: str, B: int, n: int) -> None:
    """按 V5 实测估算 n 回合要多少分钟（run_repro.sh 开头打印）。"""
    p = OUT / f"timing_raw_{model}.csv"
    rows = []
    if p.exists():
        with open(p, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if int(r["B"]) == B]
    if not rows:
        print(f"{model}: 没有 B={B} 的实测耗时（先跑 scripts/bench_timing.sh），无法估算")
        return
    r = {key: (float(v) if key != "model" else v) for key, v in rows[0].items()}
    r["B"] = B
    s = est_episode_s(model, k if k == "te" else int(k), r) * n
    print(f"{model}: k={k} B={B} {n} 回合 预计约 {s / 60:.0f} 分钟（V5 估算，每批按跑满 {T[model]} 步计，偏保守）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["measure", "estimate", "eta"])
    ap.add_argument("model", nargs="?", choices=["act", "dp"])
    ap.add_argument("--Bs", default="1,10,25,50")
    ap.add_argument("--k", default=None)
    ap.add_argument("--B", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--variant", default="", help="例如 _cpu（CPU 渲染，先 export CHUNK_RENDER=cpu）")
    a = ap.parse_args()
    if a.cmd == "measure":
        measure(a.model, [int(x) for x in a.Bs.split(",")], a.variant)
    elif a.cmd == "eta":
        eta(a.model, a.k, a.B, a.n)
    else:
        estimate()


if __name__ == "__main__":
    main()
