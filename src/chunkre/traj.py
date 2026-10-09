"""轨迹记录（--log-traj）与画面导出（--save-frames），T3。

两者都只**读取**已有数据（发给环境的动作、观测里的 agent_pos 和相机图像、仿真物理状态），
不额外渲染、不消耗随机数、不改变任何张量，所以打开与否不影响结果（T3 在 CPU 渲染下逐回合验证）。

轨迹文件：<out>/traj/batch_<批号>.npz，每批一个，数组的前两维是 (循环步 t, 环境 i)：
  seeds (B,)  steps (B,)            该批各环境的种子 / 回合步数
  alive (T, B) bool                  该步该环境的回合是否还在进行（之后的值都是 nan）
  action (T, B, A) float32           第 t 步实际发给环境的动作（环境单位）
  policy_action (T, B, A) float32    只有开了改动作的扰动（T5 action_delay，d>0）：第 t 步策略给出的动作（on_action 之前）
  agent_pos (T, B, A') float32       第 t 步策略看到的观测里的 agent_pos（aloha：14 维关节位置，夹爪为归一化开合度）
  repredict (T,) bool                第 t 步策略是否重新预测了动作块（同一批所有环境同时）；
                                     adaptive 模式（T11a）= 这一步调用了策略（候选块，不一定换计划）
  chunk_idx (T,) int16               第 t 步执行的是当前动作块里的第几个动作（0 起）；te 模式为 -1（多块加权平均）；
                                     adaptive 模式为 -1（各环境计划不同步，见 <out>/adaptive/batch_*.npz 的 plan_j）
  chunk_offsets (J,) int16           下面 chunk_sel 记录的块内位置（0 起）
  chunk_sel (T, B, J, A) float32     重新预测那一步，新动作块在 chunk_offsets 位置上的动作；其他步为 nan
                                     （动作块 = predict_action_chunk 的输出：ACT 为完整 100 步；DP 为将要执行的 k 步）
  cube_pos (T, B, 3) float32         只有 aloha：第 t 步（执行动作前）方块的位置 xyz（米），读自 physics.data.qpos[16:19]
  cube_pos_after (T, B, 3) float32   只有 aloha 且开了改仿真状态的扰动（T4 push）：第 t 步 on_state 钩子之后、
                                     env.step 之前的方块位置。推的那一步 cube_pos_after - cube_pos = 实际推的位移，其他步两者相同。
  finger_tip (T, B, 4, 3) float32    只有 aloha（T13）：第 t 步（执行动作前）4 个手指的指尖位置 xyz（米），顺序见 FINGER_GEOMS
                                     （左臂左指、左臂右指、右臂左指、右臂右指）。指尖 = 手指网格在手指 body 坐标系里沿 +x
                                     （夹爪朝向）最远 5 mm 内顶点的平均点（只在第一次读时由模型算一次），每步用 body 的 xpos / xmat
                                     变换到世界坐标。只读 physics.data，不调用 forward / step。
  block_pose (T, B, 3) float32       只有 PushT（T8b）：第 t 步（执行动作前）T 形块的 pymunk body 位置 x、y（PushT 坐标，0–512）
                                     和角度（弧度，未取模），读自 env.unwrapped.block
  block_pose_after (T, B, 3) float32 只有 PushT 且开了改仿真状态的扰动（T8b push）：同 cube_pos_after
注意：记录的是“执行第 t 步动作之前”的状态，与该步动作配对；回合结束那一步之后的状态不记录
（gymnasium 0.29 同一步自动 reset，结束后读到的是新回合的状态）。

画面：<out>/frames/seed<种子>/step<步>.png（每 frame_every 步一张，取观测里已有的相机图像，aloha 为 top 相机，
PushT 为 pixels），以及 <out>/frames/seed<种子>.gif（由同一批 PNG 缩小合成，保证 <= 2 MB）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CHUNK_OFFSETS = (0, 1, 4, 9, 19, 49, 99)   # 第 1、2、5、10、20、50、100 个动作（超出块长的位置自动丢掉）
GIF_MAX_BYTES = 2 * 1024 * 1024


def cube_positions(env) -> np.ndarray | None:
    """aloha：各环境方块位置 (B, 3)；不是 aloha（没有 dm_control physics）时返回 None。只读。"""
    out = []
    for e in env.envs:
        u = e.unwrapped
        if not hasattr(u, "_env") or not hasattr(u._env, "physics"):
            return None
        out.append(np.array(u._env.physics.data.qpos[16:19], dtype=np.float64))
    return np.stack(out)


FINGER_GEOMS = ("vx300s_left/10_left_gripper_finger", "vx300s_left/10_right_gripper_finger",
                "vx300s_right/10_left_gripper_finger", "vx300s_right/10_right_gripper_finger")
TIP_BAND = 0.005   # 指尖 = 沿 +x 最远 5 mm 内的网格顶点平均
_TIP_CACHE: dict = {}


def finger_tip_offsets(physics) -> tuple[list[int], np.ndarray]:
    """4 个手指：(body id 列表, 指尖在各自 body 坐标系里的位置 (4, 3))。只读模型常量；同一模型只算一次。"""
    m = physics.model
    key = (int(m.ngeom), int(m.nbody), int(m.nmeshvert))
    if key in _TIP_CACHE:
        return _TIP_CACHE[key]
    bodies, offs = [], []
    for name in FINGER_GEOMS:
        g = m.name2id(name, "geom")
        mid = int(m.geom_dataid[g])
        v = np.asarray(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], dtype=np.float64)
        q = np.asarray(m.geom_quat[g], dtype=np.float64)
        R = _quat_to_mat(q)
        vb = v @ R.T + np.asarray(m.geom_pos[g], dtype=np.float64)       # 网格顶点 -> 手指 body 坐标系
        sel = vb[vb[:, 0] >= vb[:, 0].max() - TIP_BAND]
        bodies.append(int(m.geom_bodyid[g]))
        offs.append(sel.mean(axis=0))
    _TIP_CACHE[key] = (bodies, np.stack(offs))
    return _TIP_CACHE[key]


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(c) for c in q / np.linalg.norm(q))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def finger_tips(env) -> np.ndarray | None:
    """aloha：各环境 4 个指尖的世界坐标 (B, 4, 3)（顺序见 FINGER_GEOMS）；不是 aloha 时返回 None。只读。"""
    out = []
    for e in env.envs:
        u = e.unwrapped
        if not hasattr(u, "_env") or not hasattr(u._env, "physics"):
            return None
        ph = u._env.physics
        bodies, offs = finger_tip_offsets(ph)
        xpos = np.asarray(ph.data.xpos, dtype=np.float64)
        xmat = np.asarray(ph.data.xmat, dtype=np.float64)
        out.append(np.stack([xpos[b] + xmat[b].reshape(3, 3) @ o for b, o in zip(bodies, offs)]))
    return np.stack(out)


def block_poses(env) -> np.ndarray | None:
    """PushT：各环境 T 形块位姿 (B, 3) = body 位置 x、y 和角度（弧度）；不是 PushT 时返回 None。只读。"""
    out = []
    for e in env.envs:
        u = e.unwrapped
        if not hasattr(u, "block") or not hasattr(u, "agent"):
            return None
        b = u.block
        out.append(np.array([b.position[0], b.position[1], b.angle], dtype=np.float64))
    return np.stack(out)


def object_state(env) -> tuple[str, np.ndarray] | None:
    """被操作物体的状态：aloha -> ("cube_pos", (B,3))；PushT -> ("block_pose", (B,3))；都不是 -> None。"""
    cp = cube_positions(env)
    if cp is not None:
        return "cube_pos", cp
    bp = block_poses(env)
    if bp is not None:
        return "block_pose", bp
    return None


def camera_image(obs) -> np.ndarray | None:
    """观测里已有的相机图像 (B, H, W, 3) uint8：aloha 取 pixels['top']，PushT 取 pixels。"""
    px = obs.get("pixels") if isinstance(obs, dict) else None
    if px is None:
        return None
    if isinstance(px, dict):
        px = px.get("top", next(iter(px.values())))
    return np.asarray(px)


class TrajRecorder:
    """逐步记录一批回合。log_traj=False 且 frame_seeds 为空时什么都不做。"""

    def __init__(self, out_dir: Path, batch_idx: int, seeds: list[int], log_traj: bool,
                 frame_seeds=(), frame_every: int = 10, te: bool = False):
        self.out_dir = Path(out_dir)
        self.batch_idx = batch_idx
        self.seeds = list(seeds)
        self.log_traj = bool(log_traj)
        self.frame_every = int(frame_every)
        self.frame_env = {i: s for i, s in enumerate(self.seeds) if s in set(frame_seeds or ())}
        self.te = te
        self.rows = []          # 每步一个 dict
        self.chunk_idx = -1
        self.frames = {i: [] for i in self.frame_env}   # env -> [(step, png 路径)]

    @property
    def active(self) -> bool:
        return self.log_traj or bool(self.frame_env)

    def record(self, t: int, obs, env, action: np.ndarray, alive: np.ndarray, repredicted: bool, chunk,
               policy_action: np.ndarray | None = None):
        """在执行第 t 步动作之前调用（obs 是策略这一步看到的观测）。chunk：本步新预测的完整动作块 (B, H, A)
        （CPU 上的 numpy，没有重新预测时为 None）。"""
        if self.frame_env and t % self.frame_every == 0:
            img = camera_image(obs)
            if img is not None:
                for i, s in self.frame_env.items():
                    if alive[i]:
                        self._save_png(img[i], s, t, i)
        if not self.log_traj:
            return
        if self.te:
            self.chunk_idx = -1
        else:
            self.chunk_idx = 0 if repredicted else self.chunk_idx + 1
        row = dict(action=np.asarray(action, dtype=np.float32), alive=alive.copy(), repredict=bool(repredicted),
                   chunk_idx=self.chunk_idx)
        if policy_action is not None:
            row["policy_action"] = np.asarray(policy_action, dtype=np.float32)
        if isinstance(obs, dict) and "agent_pos" in obs:
            row["agent_pos"] = np.asarray(obs["agent_pos"], dtype=np.float32)
        ob = object_state(env)
        if ob is not None:
            row[ob[0]] = ob[1].astype(np.float32)
        tips = finger_tips(env)
        if tips is not None:
            row["finger_tip"] = tips.astype(np.float32)
        if chunk is not None:
            offs = [o for o in CHUNK_OFFSETS if o < chunk.shape[1]]
            row["chunk_sel"] = np.asarray(chunk[:, offs], dtype=np.float32)
            self._offs = offs
        self.rows.append(row)

    def record_after_state(self, env):
        """扰动的 on_state 钩子之后调用：给刚记录的这一步补上方块位置（只读）。"""
        if not self.log_traj or not self.rows:
            return
        ob = object_state(env)
        if ob is not None:
            self.rows[-1][ob[0] + "_after"] = ob[1].astype(np.float32)

    def _save_png(self, img, seed, t, i):
        from PIL import Image

        d = self.out_dir / "frames" / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"step{t:04d}.png"
        Image.fromarray(np.ascontiguousarray(img)).save(p)
        self.frames[i].append((t, p))

    def finish(self, steps: np.ndarray):
        if self.log_traj and self.rows:
            self._write_npz(steps)
        for i, s in self.frame_env.items():
            if self.frames[i]:
                write_gif([p for _, p in self.frames[i]], self.out_dir / "frames" / f"seed{s}.gif")

    def _write_npz(self, steps):
        T, B = len(self.rows), len(self.seeds)
        alive = np.stack([r["alive"] for r in self.rows])

        def stack(key, shape_tail):
            arr = np.full((T, B) + shape_tail, np.nan, dtype=np.float32)
            for t, r in enumerate(self.rows):
                if key in r:
                    arr[t] = r[key]
            arr[~alive] = np.nan
            return arr

        A = self.rows[0]["action"].shape[1]
        out = dict(seeds=np.array(self.seeds), steps=np.asarray(steps), alive=alive,
                   action=stack("action", (A,)),
                   repredict=np.array([r["repredict"] for r in self.rows]),
                   chunk_idx=np.array([r["chunk_idx"] for r in self.rows], dtype=np.int16))
        if "policy_action" in self.rows[0]:
            out["policy_action"] = stack("policy_action", (A,))
        if "agent_pos" in self.rows[0]:
            out["agent_pos"] = stack("agent_pos", (self.rows[0]["agent_pos"].shape[1],))
        for key in ("cube_pos", "cube_pos_after", "block_pose", "block_pose_after"):
            if key in self.rows[0]:
                out[key] = stack(key, (3,))
        if "finger_tip" in self.rows[0]:
            out["finger_tip"] = stack("finger_tip", self.rows[0]["finger_tip"].shape[1:])
        if hasattr(self, "_offs"):
            out["chunk_offsets"] = np.array(self._offs, dtype=np.int16)
            out["chunk_sel"] = stack("chunk_sel", (len(self._offs), A))
        d = self.out_dir / "traj"
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"batch_{self.batch_idx:04d}.npz", **out)


def write_gif(png_paths, gif_path: Path, max_bytes: int = GIF_MAX_BYTES, fps: float = 5.0) -> int:
    """把 PNG 序列缩小合成为 GIF；超过 max_bytes 就继续缩小 / 隔帧抽取，返回最终字节数。"""
    from PIL import Image

    frames0 = [Image.open(p).convert("RGB") for p in png_paths]
    for width, stride in [(320, 1), (240, 1), (240, 2), (160, 2), (120, 3), (96, 4)]:
        frames = []
        for im in frames0[::stride]:
            h = round(im.height * width / im.width)
            frames.append(im.resize((width, h), Image.BILINEAR).quantize(colors=128, method=Image.MEDIANCUT))
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0,
                       duration=int(1000 * stride / fps), optimize=True)
        size = Path(gif_path).stat().st_size
        if size <= max_bytes:
            return size
    return size
