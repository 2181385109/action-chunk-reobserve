"""T2 排查：ACT 同配置两次结果不同，逐层定位不确定性来源（渲染 vs 策略推理）。

用法（WSL 内）：
  python scripts/probe_determinism.py render --B 10 --steps 60     # 固定动作序列步进，逐步对所有环境画面取哈希
  python scripts/probe_determinism.py act_infer --B 10             # 固定输入（种子 0 的随机张量）跑 ACT 前向，取输出哈希
  python scripts/probe_determinism.py gpu_load --seconds 60        # 制造 GPU 负载（并发对照用）
输出一行 JSON：每步画面哈希的总摘要、逐步哈希（便于找第一个不同的步）、各环境的 GL_RENDERER。
只用调试种子 0..B-1。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def render(B: int, steps: int) -> dict:
    import numpy as np
    import OpenGL.GL as gl

    from chunkre.runner import make_vec_env
    from chunkre.seed_guard import check_seeds

    seeds = list(range(B))
    check_seeds(seeds)
    env = make_vec_env("act", B)
    obs, _ = env.reset(seed=seeds)
    rends = []
    for e in env.envs:
        with e.unwrapped._env.physics.contexts.gl.make_current():
            r = gl.glGetString(gl.GL_RENDERER)
        rends.append(r.decode())
    rng = np.random.default_rng(0)
    base = obs["agent_pos"].copy()
    per_step, h_all = [], hashlib.md5()
    for t in range(steps):
        act = base + rng.normal(scale=0.02, size=base.shape)   # 固定（与进程无关）的动作序列
        obs, *_ = env.step(act.astype(np.float32))
        px = obs["pixels"]["top"]
        h = hashlib.md5(np.ascontiguousarray(px).tobytes()).hexdigest()[:12]
        per_step.append(h)
        h_all.update(h.encode())
    qpos = [hashlib.md5(e.unwrapped._env.physics.data.qpos.tobytes()).hexdigest()[:12] for e in env.envs]
    env.close()
    return dict(mode="render", B=B, steps=steps, digest=h_all.hexdigest()[:16], renderers=sorted(set(rends)),
                per_step=per_step, final_qpos=qpos)


def act_infer(B: int) -> dict:
    import torch

    from chunkre.config import load_config
    from chunkre.runner import build_policy, set_determinism, set_global_seed

    set_determinism(True)
    cfg = load_config(None, dict(model="act", seeds="0-0", batch_size=1))
    policy, _ = build_policy(cfg)
    set_global_seed(0)
    g = torch.Generator().manual_seed(0)
    batch = {"observation.images.top": torch.rand(B, 3, 480, 640, generator=g).cuda(),
             "observation.state": (torch.rand(B, 14, generator=g) - 0.5).cuda()}
    outs = []
    with torch.inference_mode():
        for _ in range(3):
            a = policy.predict_action_chunk(dict(batch))
            outs.append(hashlib.md5(a.cpu().numpy().tobytes()).hexdigest()[:12])
    return dict(mode="act_infer", B=B, hashes=outs)


def pusht_order() -> dict:
    """gym_pusht 的 _get_coverage 遍历 self.block.shapes（pymunk 返回 set，迭代顺序随对象内存地址变化）。
    把 T 块的两个多边形按两种顺序合成几何体，比较覆盖率：差异只在浮点尾数。"""
    import gymnasium as gym
    import gym_pusht  # noqa: F401
    from gym_pusht.envs.pusht import pymunk_to_shapely

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    import math

    # 把 T 块放在目标位姿附近（部分重叠），覆盖率不为 0 才能看出顺序的影响
    env.reset(seed=2, options={"reset_to_state": [100.0, 100.0, 259.0, 253.0, math.pi / 4 + 0.07]})
    e = env.unwrapped
    shapes = list(e.block.shapes)
    goal = e.get_goal_pose_body(e.goal_pose)
    covs = []
    for order in (shapes, shapes[::-1]):
        g = pymunk_to_shapely(goal, order)
        b = pymunk_to_shapely(e.block, order)
        covs.append(g.intersection(b).area / g.area)
    return dict(mode="pusht_order", coverage_order_a=repr(covs[0]), coverage_order_b=repr(covs[1]),
                abs_diff=abs(covs[0] - covs[1]), type_of_block_shapes=type(e.block.shapes).__name__)


def gpu_load(seconds: float) -> dict:
    import torch

    x = torch.randn(4096, 4096, device="cuda")
    t0, n = time.time(), 0
    while time.time() - t0 < seconds:
        y = x @ x
        n += 1
        if n % 50 == 0:
            torch.cuda.synchronize()
    return dict(mode="gpu_load", iters=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["render", "act_infer", "gpu_load", "pusht_order"])
    ap.add_argument("--B", type=int, default=10)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.mode == "render":
        r = render(a.B, a.steps)
    elif a.mode == "act_infer":
        r = act_infer(a.B)
    elif a.mode == "pusht_order":
        r = pusht_order()
    else:
        r = gpu_load(a.seconds)
    r["tag"] = a.tag
    print("PROBE " + json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
