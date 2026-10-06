"""冒烟测试（T1）：每个模型用官方默认推理设置、在 GPU 上跑 2 个回合（种子 0、1）。

做法与 LeRobot 0.3.3 的 lerobot/scripts/eval.py 的 rollout() 一致（batch=1 的向量环境，
preprocess_observation -> policy.select_action -> env.step），只是多了计时和第一帧图像保存。
不改 LeRobot 源码，不加 k 配置/扰动（那是后续任务）。

输出：
  results/smoke/smoke_episodes.csv   一回合一行
  results/smoke/smoke.json           汇总 + 配置 + 环境版本
  results/smoke/first_obs_<model>.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_guard import check_seeds  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.envs.configs import AlohaEnv, PushtEnv  # noqa: E402
from lerobot.envs.factory import make_env  # noqa: E402
from lerobot.envs.utils import add_envs_task, preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_policy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "smoke"

MODELS = {
    "act": dict(repo="lerobot/act_aloha_sim_transfer_cube_human", env=lambda: AlohaEnv(task="AlohaTransferCube-v0")),
    "dp": dict(repo="lerobot/diffusion_pusht", env=lambda: PushtEnv()),
}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def save_png(arr: np.ndarray, path: Path) -> dict:
    from PIL import Image

    arr = np.asarray(arr)
    Image.fromarray(arr).save(path)
    return dict(
        shape=list(arr.shape), dtype=str(arr.dtype),
        pixel_mean=float(arr.mean()), pixel_std=float(arr.std()),
        n_unique_values=int(len(np.unique(arr))),
    )


def run_model(name: str, spec: dict, seeds: list[int], device: str, time_limit_s: float):
    env_cfg = spec["env"]()
    env = make_env(env_cfg, n_envs=1, use_async_envs=False)
    pcfg = PreTrainedConfig.from_pretrained(spec["repo"])
    pcfg.pretrained_path = spec["repo"]
    pcfg.device = device
    t0 = time.perf_counter()
    policy = make_policy(pcfg, env_cfg=env_cfg)
    policy.eval()
    load_s = time.perf_counter() - t0
    info = dict(
        policy_class=type(policy).__name__,
        chunk_size=getattr(pcfg, "chunk_size", None),
        horizon=getattr(pcfg, "horizon", None),
        n_action_steps=pcfg.n_action_steps,
        n_obs_steps=pcfg.n_obs_steps,
        temporal_ensemble_coeff=getattr(pcfg, "temporal_ensemble_coeff", None),
        device=str(next(policy.parameters()).device),
        load_s=load_s,
    )
    max_steps = env.call("_max_episode_steps")[0]
    rows, image_stats = [], None
    for ep, seed in enumerate(seeds):
        check_seeds([seed])
        policy.reset()
        obs, _ = env.reset(seed=[seed])
        if image_stats is None:
            px = obs["pixels"]
            img = px["top"][0] if isinstance(px, dict) else px[0]
            OUT.mkdir(parents=True, exist_ok=True)
            image_stats = save_png(img, OUT / f"first_obs_{name}.png")
        t_pol = t_env = 0.0
        n_pol_calls = 0
        steps, rewards, success, done = 0, [], False, False
        t_ep = time.perf_counter()
        while not done:
            o = preprocess_observation(obs)
            o = {k: v.to(device) for k, v in o.items()}
            o = add_envs_task(env, o)
            sync(); a = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(o)
            sync(); t_pol += time.perf_counter() - a
            action = action.cpu().numpy()
            a = time.perf_counter()
            obs, reward, terminated, truncated, inf = env.step(action)
            t_env += time.perf_counter() - a
            steps += 1
            rewards.append(float(reward[0]))
            if "final_info" in inf:
                fi = inf["final_info"][0]
                success = bool(fi["is_success"]) if fi is not None else False
            done = bool(terminated[0] or truncated[0])
            if success is False and "is_success" in inf:
                pass
        wall = time.perf_counter() - t_ep
        rows.append(dict(
            model=name, seed=seed, success=int(success), steps=steps, max_steps=int(max_steps),
            sum_reward=float(np.sum(rewards)), max_reward=float(np.max(rewards)),
            wall_s=wall, policy_s=t_pol, env_s=t_env, other_s=wall - t_pol - t_env,
        ))
        print(f"[{name}] seed={seed} success={success} steps={steps}/{max_steps} wall={wall:.1f}s "
              f"(policy {t_pol:.1f}s, env {t_env:.1f}s)", flush=True)
        if wall > time_limit_s and ep + 1 < len(seeds):
            print(f"[{name}] 单回合超过 {time_limit_s:.0f}s，只跑 1 回合。", flush=True)
            info["truncated_to_one_episode"] = True
            break
    env.close()
    return rows, info, image_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="act,dp")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--time_limit_s", type=float, default=180.0)
    ap.add_argument("--out", default="results/smoke", help="输出目录（相对仓库根）")
    args = ap.parse_args()
    global OUT
    OUT = ROOT / args.out
    seeds = [int(s) for s in args.seeds.split(",")]
    check_seeds(seeds)
    assert torch.cuda.is_available() or args.device != "cuda", "CUDA 不可用"
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows, per_model = [], {}
    for name in args.models.split(","):
        rows, info, img = run_model(name, MODELS[name], seeds, args.device, args.time_limit_s)
        all_rows += rows
        per_model[name] = dict(repo=MODELS[name]["repo"], policy=info, first_obs_image=img)

    cols = list(all_rows[0].keys())
    with open(OUT / "smoke_episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(all_rows)

    summary = {}
    for name in per_model:
        r = [x for x in all_rows if x["model"] == name]
        summary[name] = dict(
            n_episodes=len(r), n_success=int(sum(x["success"] for x in r)),
            mean_steps=float(np.mean([x["steps"] for x in r])),
            mean_wall_s=float(np.mean([x["wall_s"] for x in r])),
            mean_policy_s=float(np.mean([x["policy_s"] for x in r])),
            mean_env_s=float(np.mean([x["env_s"] for x in r])),
            mean_max_reward=float(np.mean([x["max_reward"] for x in r])),
        )
    vers = {}
    for p in ["lerobot", "torch", "torchvision", "numpy", "gymnasium", "gym-aloha", "gym-pusht", "mujoco", "dm-control"]:
        try:
            vers[p] = metadata.version(p)
        except Exception:
            vers[p] = None
    out = dict(
        config=dict(seeds=seeds, device=args.device, MUJOCO_GL=os.environ.get("MUJOCO_GL"),
                    HF_ENDPOINT=os.environ.get("HF_ENDPOINT"), note="官方默认推理设置，未改 n_action_steps"),
        versions=vers, python=platform.python_version(),
        cuda=dict(available=torch.cuda.is_available(), torch_cuda=torch.version.cuda,
                  device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        models=per_model, summary=summary,
        episodes_csv=f"{args.out}/smoke_episodes.csv",
    )
    with open(OUT / "smoke.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
