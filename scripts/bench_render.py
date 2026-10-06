"""渲染后端 / 推理耗时小基准（T1，用于 docs/ENVIRONMENT.md）。

用法：MUJOCO_GL=egl python scripts/bench_render.py [aloha|pusht|dp_infer]
- aloha：创建 AlohaTransferCube-v0，reset(seed=0)，用零动作（保持初始位姿）步进 N 步，报告每步耗时。
- dp_infer：加载 diffusion_pusht，测一次动作块预测（100 步去噪）的耗时（GPU）。
只用调试种子 0。不是评估，不产生成功率。
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from seed_guard import check_seeds  # noqa: E402

which = sys.argv[1] if len(sys.argv) > 1 else "aloha"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
check_seeds([0])

if which in ("aloha", "pusht"):
    import gymnasium as gym
    import gym_aloha  # noqa: F401
    import gym_pusht  # noqa: F401

    if which == "aloha":
        env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array",
                       max_episode_steps=400)
    else:
        env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array",
                       max_episode_steps=300)
    obs, _ = env.reset(seed=0)
    a = env.action_space.sample() * 0 + (obs["agent_pos"] if "agent_pos" in obs else 0)
    t = time.perf_counter()
    for _ in range(N):
        obs, *_ = env.step(a)
    dt = time.perf_counter() - t
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')} {which}: {N} steps in {dt:.2f}s = {1000 * dt / N:.1f} ms/step")
elif which == "dp_infer":
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import PushtEnv
    from lerobot.policies.factory import make_policy

    repo = "lerobot/diffusion_pusht"
    cfg = PreTrainedConfig.from_pretrained(repo)
    cfg.pretrained_path = repo
    cfg.device = "cuda"
    pol = make_policy(cfg, env_cfg=PushtEnv()).eval()
    obs = {
        "observation.image": torch.rand(1, 3, 96, 96, device="cuda"),
        "observation.state": torch.rand(1, 2, device="cuda") * 400,
    }
    ts = []
    for i in range(6):
        pol.reset()
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.inference_mode():
            for _ in range(cfg.n_action_steps):  # 第一次调用预测，其余从队列取；n_obs_steps=2 需要先填满观测队列
                pol.select_action(obs)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    print("DP: 一个 8 步动作块（含 1 次预测）耗时(s)：", [round(x, 3) for x in ts])
    print("params(M):", sum(p.numel() for p in pol.parameters()) / 1e6)
