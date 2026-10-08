"""统一评估核心：批量环境逐步循环（参照 lerobot 0.3.3 lerobot/scripts/eval.py 的 rollout），可配置执行长度 k。

和 LeRobot 的 rollout 保持一致的地方（为了能与 LeRobot 自带评估逐回合对齐，见 V3）：
- 用 lerobot 的 make_env 建 gymnasium SyncVectorEnv（B 个环境），一次前向给所有环境出动作；
- 所有环境一起步进，直到全部结束；先结束的环境会被 gymnasium 自动 reset 并继续被步进，
  但它之后的数据被 done 掩码丢掉（LeRobot 也是这样）。这样策略的批大小、随机数消耗和 LeRobot 完全相同；
- 观测预处理 preprocess_observation + add_envs_task，torch.inference_mode 下调用 policy.select_action。

和 LeRobot 不同的地方：
- 每批开始先用该批第一个种子设定 random/numpy/torch 的全局种子（LeRobot 只在评估开始时设一次）；
- 逐回合统计只算到该回合结束那一步（LeRobot 的掩码多算了结束后的一步，见 docs/notes_zh/02）；
- 额外统计策略调用次数、动作平滑度、墙钟时间；可设 max_steps 截断（只用于检查）。
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import platform
import socket
import subprocess
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from .config import MODELS, TE_COEFF, EvalConfig, check_local_policy_dir, validate_k
from .metrics import EPISODE_COLUMNS, csv_columns, motion_metrics, summarize_csv
from .perturb import Perturbation, make_perturbation
from .seed_guard import check_seeds
from .traj import TrajRecorder

ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------------------------
# 全局随机性 / 确定性
# ----------------------------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """与 lerobot.utils.random_utils.set_seed 相同：random、numpy、torch（含所有 CUDA 设备）。"""
    from lerobot.utils.random_utils import set_seed

    set_seed(int(seed))


def set_determinism(on: bool) -> dict:
    """设定 torch 后端开关，返回实际设置（写进 config.json）。

    - cudnn.benchmark=False（benchmark 会按运行时测速挑卷积算法，可能每次不同）；
    - cudnn.deterministic=on；
    - on 时 torch.use_deterministic_algorithms(True)（需要 CUBLAS_WORKSPACE_CONFIG，env.sh 已设）；
    - matmul TF32 与 LeRobot eval_main 一致设为 True（TF32 本身是确定的，只是精度不同）。
    """
    if on:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(on)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(bool(on))
    return dict(
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        use_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------------------------
# 策略 / 环境构建
# ----------------------------------------------------------------------------------------------
def make_env_cfg(model: str):
    from lerobot.envs.factory import make_env_config

    spec = MODELS[model]
    return make_env_config(spec["env_type"], **spec["env_kwargs"])


def build_policy(cfg: EvalConfig):
    """加载策略（hub 预训练；cfg.policy_path 不为 None 时从本地检查点目录，T10a），按 cfg 设置 n_action_steps / 时间集成。
    返回 (policy, policy_info)。"""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy

    spec = MODELS[cfg.model]
    source = spec["repo"]
    if cfg.policy_path is not None:
        source = str(Path(cfg.policy_path).resolve())
        check_local_policy_dir(source, cfg.model)
    pcfg = PreTrainedConfig.from_pretrained(source)
    pcfg.pretrained_path = source
    pcfg.device = cfg.device
    hub_default_k = int(pcfg.n_action_steps)
    if hub_default_k != spec["default_k"]:
        raise RuntimeError(f"hub config 的 n_action_steps={hub_default_k} 与预期默认 {spec['default_k']} 不符")
    validate_k(cfg.model, cfg.mode, cfg.k_effective, pcfg)
    if cfg.mode == "te":
        pcfg.temporal_ensemble_coeff = TE_COEFF
        pcfg.n_action_steps = 1
    elif cfg.k is not None:
        pcfg.n_action_steps = int(cfg.k)
    # cfg.k 为 None 时完全不碰 n_action_steps（V2 检查“显式传默认值”与“不传”一致）
    policy = make_policy(pcfg, env_cfg=make_env_cfg(cfg.model))
    policy.eval()
    pc = policy.config
    info = dict(
        repo=source, policy_class=type(policy).__name__,
        n_action_steps=int(pc.n_action_steps), hub_default_n_action_steps=hub_default_k,
        n_obs_steps=int(pc.n_obs_steps),
        chunk_size=getattr(pc, "chunk_size", None), horizon=getattr(pc, "horizon", None),
        temporal_ensemble_coeff=getattr(pc, "temporal_ensemble_coeff", None),
        num_inference_steps=getattr(pc, "num_inference_steps", None),
        noise_scheduler_type=getattr(pc, "noise_scheduler_type", None),
        k_upper_bound=(int(pc.chunk_size) if cfg.model == "act"
                       else int(pc.horizon) - int(pc.n_obs_steps) + 1),
        use_amp=bool(getattr(pc, "use_amp", False)),
        n_params_M=round(sum(p.numel() for p in policy.parameters()) / 1e6, 2),
    )
    if cfg.model == "dp":
        info["num_inference_steps_effective"] = int(policy.diffusion.num_inference_steps)
    return policy, info


def make_vec_env(model: str, n_envs: int):
    from lerobot.envs.factory import make_env

    return make_env(make_env_cfg(model), n_envs=n_envs, use_async_envs=False)


class CallCounter:
    """包装 policy.predict_action_chunk（只在我们的进程里替换实例属性，不改 LeRobot 源码），统计重新预测次数。

    ACT 与 DP 的 select_action 都在动作队列空了（或时间集成时每一步）时调用 self.predict_action_chunk，
    所以每调用一次 = 策略实际重新预测了一个动作块（对批里所有环境同时）。
    keep_last=True 时（轨迹记录用）把最近一次预测的完整动作块复制一份到 CPU（self.last，numpy），只读不改。
    """

    def __init__(self, policy):
        self.n = 0
        self.t = 0.0
        self.keep_last = False
        self.last = None
        self._orig = policy.predict_action_chunk

        def wrapped(*args, **kwargs):
            self.n += 1
            out = self._orig(*args, **kwargs)
            if self.keep_last:
                self.last = out.detach().to("cpu", copy=True).float().numpy()
            return out

        policy.predict_action_chunk = wrapped
        self._policy = policy

    def remove(self):
        self._policy.predict_action_chunk = self._orig


# ----------------------------------------------------------------------------------------------
# 渲染器
# ----------------------------------------------------------------------------------------------
def gl_renderer_of(env) -> str:
    """读取 aloha（dm_control）环境实际使用的 GL_RENDERER；PushT 用 pygame，不经 OpenGL。"""
    e = env.envs[0].unwrapped
    if not hasattr(e, "_env") or not hasattr(e._env, "physics"):
        return "n/a (pygame 渲染，不经过 OpenGL)"
    import OpenGL.GL as gl

    with e._env.physics.contexts.gl.make_current():
        r = gl.glGetString(gl.GL_RENDERER)
    return r.decode() if isinstance(r, bytes) else str(r)


def check_renderer(model: str, renderer: str, allow_cpu_render: bool) -> None:
    if model == "act" and "D3D12" not in renderer and not allow_cpu_render:
        raise RuntimeError(
            f"aloha 实际渲染器是 {renderer!r}，不是 D3D12（GPU）。全研究固定 GPU 渲染；"
            "请先 source scripts/env.sh（CHUNK_RENDER=gpu），或显式传 --allow-cpu-render。"
        )


# ----------------------------------------------------------------------------------------------
# 一批回合
# ----------------------------------------------------------------------------------------------
def run_batch(env, policy, counter: CallCounter, seeds: list[int], device: str,
              max_steps: int | None = None, after_reset=None,
              recorder: TrajRecorder | None = None,
              perturb: Perturbation | None = None) -> tuple[list[dict], dict]:
    """用 len(seeds) 个环境同时跑一批回合。返回 (逐回合统计列表, 批统计)。

    逐回合统计：
      success      该回合在结束那一步是否成功（来自 final_info["is_success"]）
      steps        该回合实际执行的环境步数（含结束那一步）
      capped       1 = 达到 max_steps 截断时还没结束（只在检查里出现）
      sum_reward / max_reward  该回合各步奖励的和 / 最大值（只算到结束那一步）
      n_policy_calls  该环境存活期间策略重新预测动作块的次数
      smoothness   相邻两步动作差的 L2 范数的平均值（见 metrics.action_smoothness）
      path_length / jerk / reversal_rate  T3 运动指标（见 metrics.path_length / jerk / reversal_rate）
      wall_s       从这批 reset 完成到该回合结束的墙钟秒数（同批环境并行，所以各回合有重叠）
    recorder：轨迹记录 / 画面导出（chunkre.traj），只读观测、动作和物理状态，不影响结果。
    perturb：扰动（chunkre.perturb）。第 t 步顺序：obs -> on_obs -> 策略 -> on_action -> 记录 -> on_state -> env.step。
      None 或 kind=none 时三个钩子都是原样返回。逐回合扰动记录合并进返回的逐回合统计（PERTURB_COLUMNS）。
    """
    from lerobot.envs.utils import add_envs_task, preprocess_observation

    B = len(seeds)
    assert env.num_envs == B
    set_global_seed(seeds[0])
    policy.reset()
    obs, _ = env.reset(seed=list(seeds))
    pert = perturb if perturb is not None else Perturbation()
    pert.begin_batch(env, seeds)       # 只用每回合独立的随机数，不碰全局随机状态
    if after_reset is not None:
        after_reset(env)
    T_env = int(env.call("_max_episode_steps")[0])
    limit = min(T_env, int(max_steps)) if max_steps else T_env

    done = np.zeros(B, dtype=bool)
    success = np.zeros(B, dtype=bool)
    steps = np.zeros(B, dtype=np.int64)
    sum_r = np.zeros(B, dtype=np.float64)
    max_r = np.full(B, -np.inf)
    calls = np.zeros(B, dtype=np.int64)
    diff_sum = np.zeros(B, dtype=np.float64)
    diff_n = np.zeros(B, dtype=np.int64)
    ep_wall = np.zeros(B, dtype=np.float64)
    prev_action = None
    ep_actions = [[] for _ in range(B)]   # 各环境存活期间实际执行的动作（算运动指标）
    rec = recorder if (recorder is not None and recorder.active) else None
    counter.keep_last = bool(rec is not None and rec.log_traj)
    t_pol = t_env = 0.0
    is_cuda = torch.device(device).type == "cuda"

    t0 = time.perf_counter()
    step = 0
    while not done.all() and step < limit:
        alive = ~done
        obs = pert.on_obs(step, obs, alive)
        o = preprocess_observation(obs)
        o = {key: o[key].to(device, non_blocking=is_cuda) for key in o}
        o = add_envs_task(env, o)

        n0 = counter.n
        _sync()
        ta = time.perf_counter()
        with torch.inference_mode():
            action = policy.select_action(o)
        action = action.to("cpu").numpy()
        t_pol += time.perf_counter() - ta
        assert action.ndim == 2, "动作应为 (batch, action_dim)"
        policy_action = action
        action = pert.on_action(step, action, alive)
        if counter.n > n0:
            calls[alive] += counter.n - n0
        if prev_action is not None:
            d = np.linalg.norm(action.astype(np.float64) - prev_action, axis=1)
            diff_sum[alive] += d[alive]
            diff_n[alive] += 1
        prev_action = action.astype(np.float64)
        for i in np.flatnonzero(alive):
            ep_actions[i].append(action[i])
        if rec is not None:
            repred = counter.n > n0
            rec.record(step, obs, env, action, alive, repred, counter.last if repred else None,
                       policy_action=policy_action if pert.touches_action else None)
        pert.on_state(step, env, alive)
        if rec is not None and pert.touches_state:
            rec.record_after_state(env)

        ta = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        t_env += time.perf_counter() - ta

        if "final_info" in info:
            succ_now = np.array([bool(fi["is_success"]) if fi is not None else False
                                 for fi in info["final_info"]])
        else:
            succ_now = np.zeros(B, dtype=bool)
        reward = np.asarray(reward, dtype=np.float64)
        sum_r[alive] += reward[alive]
        max_r[alive] = np.maximum(max_r[alive], reward[alive])
        success[alive] |= succ_now[alive]
        steps[alive] += 1
        ended = alive & (np.asarray(terminated) | np.asarray(truncated))
        now = time.perf_counter() - t0
        ep_wall[ended] = now
        done |= ended
        step += 1

    batch_wall = time.perf_counter() - t0
    capped = ~done
    ep_wall[capped] = batch_wall
    smooth = np.where(diff_n > 0, diff_sum / np.maximum(diff_n, 1), np.nan)
    pert.end_batch(steps)
    if rec is not None:
        rec.finish(steps)
    counter.keep_last = False
    eps = []
    for i in range(B):
        acts = np.asarray(ep_actions[i], dtype=np.float64).reshape(len(ep_actions[i]), -1)
        eps.append(dict(
            env_idx=i, seed=int(seeds[i]), success=int(success[i]), steps=int(steps[i]),
            max_steps=T_env, capped=int(capped[i]),
            sum_reward=float(sum_r[i]), max_reward=float(max_r[i]),
            n_policy_calls=int(calls[i]), smoothness=float(smooth[i]), wall_s=float(ep_wall[i]),
            **motion_metrics(acts), **pert.episode_info(i),
        ))
    batch = dict(batch_wall_s=batch_wall, batch_policy_s=t_pol, batch_env_s=t_env, n_loop_steps=step)
    return eps, batch


# ----------------------------------------------------------------------------------------------
# 整个评估（含断点续跑）
# ----------------------------------------------------------------------------------------------
def _versions() -> dict:
    out = {}
    for p in ["lerobot", "torch", "torchvision", "numpy", "gymnasium", "gym-aloha", "gym-pusht",
              "mujoco", "dm-control", "diffusers", "pygame"]:
        try:
            out[p] = metadata.version(p)
        except Exception:
            out[p] = None
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def _read_done_batches(csv_path: Path, cfg: EvalConfig) -> set[int]:
    """读已有 CSV，返回已完整写入的批次号；不完整的批次行会被删掉（重写 CSV，保持原表头）。"""
    if not csv_path.exists():
        return set()
    header = csv_columns(csv_path)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    batches = cfg.batches()
    by_b: dict[int, list[dict]] = {}
    for r in rows:
        by_b.setdefault(int(r["batch_idx"]), []).append(r)
    ok = set()
    for b, rs in by_b.items():
        if 0 <= b < len(batches) and sorted(int(r["seed"]) for r in rs) == sorted(batches[b]):
            ok.add(b)
    keep = [r for r in rows if int(r["batch_idx"]) in ok]
    if len(keep) != len(rows):
        print(f"[resume] 删除 {len(rows) - len(keep)} 行不完整批次的结果", flush=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header or EPISODE_COLUMNS)
            w.writeheader()
            w.writerows(keep)
    return ok


# 只影响“额外记录什么”、不影响结果的字段：不进配置指纹（T2 写出的旧 config.json 也没有它们，照样能续跑）
NON_RESULT_FIELDS = ("name", "log_traj", "save_frames", "frame_every")


def _fingerprint(cfg: EvalConfig) -> dict:
    d = cfg.to_dict()
    for key in NON_RESULT_FIELDS:
        d.pop(key, None)
    if not cfg.perturb_active:      # 无扰动：指纹与 T2/T3 写出的旧 config.json 相同
        d.pop("perturb", None)
    if cfg.policy_path is None:     # hub 模型：指纹与 T10a 之前写出的 config.json 相同
        d.pop("policy_path", None)
    return d


def run_eval(cfg: EvalConfig, out_dir: Path, policy_and_info=None, log=print) -> dict:
    """跑完 cfg 指定的所有批次，写 config.json / episodes.csv / summary.json，返回 summary。

    断点续跑：out_dir 里已有相同配置的 episodes.csv 时，已完整写入的批次直接跳过。
    每批开始都用该批第一个种子重设全局种子，所以续跑结果与一次跑完逐回合相同。
    """
    check_seeds(cfg.seeds, model=cfg.model)
    if cfg.policy_path is not None:
        # 解析符号链接（checkpoints/last -> 005000）：指纹里记实际检查点，“last”以后指向别的检查点时不会续写进同一结果
        cfg.policy_path = str(Path(cfg.policy_path).resolve())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path, csv_path, sum_path = out_dir / "config.json", out_dir / "episodes.csv", out_dir / "summary.json"

    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if old.get("eval_config_fingerprint") != _fingerprint(cfg):
            raise RuntimeError(f"{out_dir} 里已有不同配置的结果；换一个输出目录或手动删除后再跑。")
    done_batches = _read_done_batches(csv_path, cfg) if cfg_path.exists() else set()
    if not cfg_path.exists() and csv_path.exists():
        raise RuntimeError(f"{csv_path} 存在但没有 config.json，无法确认配置，拒绝续跑。")

    torch_flags = set_determinism(cfg.deterministic)
    if policy_and_info is None:
        policy_and_info = build_policy(cfg)
    policy, pinfo = policy_and_info
    counter = CallCounter(policy)
    perturb = make_perturbation(cfg.perturb, cfg.model)

    # 续跑 T2 写出的旧格式 CSV（没有 T3 运动指标列）时沿用旧表头，新批次也按旧表头写（运动指标不写）
    columns = csv_columns(csv_path) or EPISODE_COLUMNS
    unknown = set(columns) - set(EPISODE_COLUMNS)
    if unknown:
        raise RuntimeError(f"{csv_path} 的表头有未知列 {sorted(unknown)}，拒绝续跑。")

    batches = cfg.batches()
    todo = [b for b in range(len(batches)) if b not in done_batches]
    log(f"[{cfg.name or cfg.model}] model={cfg.model} mode={cfg.mode} k={cfg.k_effective} B={cfg.batch_size} "
        f"扰动={perturb.kind}:{perturb.strength:g} 批次 {len(batches)}，已完成 {len(done_batches)}，待跑 {len(todo)}")
    if cfg.perturb_active and not set(c for c in columns if c.startswith("perturb_")):
        raise RuntimeError(f"{csv_path} 是旧格式（没有扰动列），不能写扰动结果。")

    renderer = {}
    runtime = dict(
        started=_dt.datetime.now().isoformat(timespec="seconds"), host=socket.gethostname(),
        git_commit=_git_commit(), python=platform.python_version(), versions=_versions(),
        cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        torch_flags=torch_flags,
        env_vars={k: os.environ.get(k) for k in ["MUJOCO_GL", "PYOPENGL_PLATFORM", "GALLIUM_DRIVER",
                                                  "MESA_D3D12_DEFAULT_ADAPTER_NAME", "CHUNK_RENDER"]},
    )

    def write_config():
        prev = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        runs = prev.get("runs", [])
        runs.append(dict(runtime, gl_renderer=renderer.get("gl")))
        conf = dict(eval_config_fingerprint=_fingerprint(cfg), name=cfg.name, policy=pinfo,
                    gl_renderer=prev.get("gl_renderer") or renderer.get("gl"), runs=runs,
                    columns=columns, perturbation=perturb.describe(),
                    logging=dict(log_traj=cfg.log_traj, save_frames=cfg.save_frames, frame_every=cfg.frame_every))
        if prev.get("gl_renderer") and renderer.get("gl") and prev["gl_renderer"] != renderer["gl"]:
            raise RuntimeError(f"续跑时渲染器变了：{prev['gl_renderer']} -> {renderer['gl']}")
        cfg_path.write_text(json.dumps(conf, indent=2, ensure_ascii=False), encoding="utf-8")

    def after_reset(env):
        if "gl" not in renderer:
            renderer["gl"] = gl_renderer_of(env)
            check_renderer(cfg.model, renderer["gl"], cfg.allow_cpu_render)
            write_config()
            log(f"[render] GL_RENDERER = {renderer['gl']}")

    if todo:
        env = make_vec_env(cfg.model, cfg.batch_size)
        new_file = not csv_path.exists()
        t_start = time.perf_counter()
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
                if new_file:
                    w.writeheader()
                for j, b in enumerate(todo):
                    rec = TrajRecorder(out_dir, b, batches[b], cfg.log_traj, cfg.save_frames, cfg.frame_every,
                                       te=cfg.mode == "te")
                    eps, bstat = run_batch(env, policy, counter, batches[b], cfg.device, cfg.max_steps,
                                           after_reset=after_reset, recorder=rec, perturb=perturb)
                    for e in eps:
                        w.writerow(dict(e, batch_idx=b, model=cfg.model, mode=cfg.mode, k=cfg.k_effective,
                                        batch_wall_s=bstat["batch_wall_s"],
                                        batch_policy_s=bstat["batch_policy_s"],
                                        batch_env_s=bstat["batch_env_s"]))
                    f.flush()
                    os.fsync(f.fileno())
                    el = time.perf_counter() - t_start
                    eta = el / (j + 1) * (len(todo) - j - 1)
                    ns = sum(e["success"] for e in eps)
                    log(f"[{cfg.name or cfg.model}] 批 {b + 1}/{len(batches)} 种子 {batches[b][0]}–{batches[b][-1]} "
                        f"成功 {ns}/{len(eps)} 用时 {bstat['batch_wall_s']:.1f}s（策略 {bstat['batch_policy_s']:.1f}s，"
                        f"环境 {bstat['batch_env_s']:.1f}s） 预计还需 {eta / 60:.1f} 分钟")
        finally:
            env.close()

    counter.remove()
    summary = dict(
        name=cfg.name, model=cfg.model, mode=cfg.mode, k=cfg.k_effective, batch_size=cfg.batch_size,
        seeds=f"{cfg.seeds[0]}..{cfg.seeds[-1]} (n={len(cfg.seeds)})", max_steps=cfg.max_steps,
        metrics=summarize_csv(csv_path),
        note="metrics 由 episodes.csv 经 chunkre.metrics.summarize_rows 计算；scripts/summarize.py 可重算",
    )
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
