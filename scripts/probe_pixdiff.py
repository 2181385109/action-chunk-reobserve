"""T2 排查：量化 D3D12（GPU）渲染画面的不确定性有多大。

  python scripts/probe_pixdiff.py dump /tmp/frames_a.npy     # aloha B=1 种子 0：首帧 + 固定动作 20 步的画面；
                                                             # 并在同一进程里对同一状态重复渲染 5 次
  python scripts/probe_pixdiff.py compare a.npy b.npy c.npy --out results/t2_checks/probe_pixdiff.json
一键：bash scripts/probe_pixdiff.sh
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def dump(out: str) -> None:
    from chunkre.runner import gl_renderer_of, make_vec_env
    from chunkre.seed_guard import check_seeds

    check_seeds([0])
    env = make_vec_env("act", 1)
    obs, _ = env.reset(seed=[0])
    phys = env.envs[0].unwrapped._env.physics
    frames = [obs["pixels"]["top"][0].copy()]
    rep = [phys.render(height=480, width=640, camera_id="top") for _ in range(5)]
    rng = np.random.default_rng(0)
    base = obs["agent_pos"].copy()
    for _ in range(20):
        obs, *_ = env.step((base + rng.normal(scale=0.02, size=base.shape)).astype(np.float32))
        frames.append(obs["pixels"]["top"][0].copy())
    np.save(out, np.stack(frames))
    meta = dict(renderer=gl_renderer_of(env),
                same_state_rerender_equal_to_first=[bool(np.array_equal(rep[0], r)) for r in rep],
                same_state_rerender_n_diff_values=[int((rep[0] != r).sum()) for r in rep])
    Path(out + ".json").write_text(json.dumps(meta))
    env.close()


def compare(files: list[str], out: str) -> None:
    f = {Path(x).stem: np.load(x).astype(int) for x in files}
    meta = {k: json.loads(Path(x + ".json").read_text()) for k, x in zip(f, files)}
    names = list(f)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = np.abs(f[names[i]] - f[names[j]])
            pairs.append(dict(a=names[i], b=names[j],
                              n_values_differ_per_step=[int((d[t] > 0).sum()) for t in range(len(d))],
                              max_abs_diff=int(d.max()), total_values_per_frame=int(d[0].size)))
    res = dict(what="aloha 种子 0、B=1，GPU(D3D12) 渲染，三个独立进程的首帧+20 步画面逐像素比较", runs=meta, pairs=pairs)
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for p in pairs:
        print(p["a"], p["b"], "每步不同的像素值个数", p["n_values_differ_per_step"], "最大差", p["max_abs_diff"],
              "/ 每帧", p["total_values_per_frame"])
    for k, m in meta.items():
        print(k, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "compare"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "results/t2_checks/probe_pixdiff.json"))
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(a.files[0])
    else:
        compare(a.files, a.out)


if __name__ == "__main__":
    main()
