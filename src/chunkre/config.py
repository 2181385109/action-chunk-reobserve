"""评估配置：读 YAML、命令行覆盖、校验执行长度 k。

执行长度 k = LeRobot 0.3.3 policy config 的 n_action_steps（策略一次预测一串动作后，实际执行前 k 步再重新观察）。
上限（从 0.3.3 源码确认）：
- ACT：k <= chunk_size（=100）。configuration_act.py 的 __post_init__ 会检查，但我们在加载预训练 config 后
  直接改属性，不会再触发 __post_init__，所以这里自己检查。
- DP：k <= horizon - n_obs_steps + 1（=16-2+1=15）。见 modeling_diffusion.py select_action 的 docstring；
  generate_actions 里 actions[:, n_obs_steps-1 : n_obs_steps-1+k] 超出 horizon 时会被**静默截短**而不报错，
  所以必须自己检查。
模式：
- fixed：固定 k。
- te：ACT 的时间集成（temporal_ensemble_coeff=0.01，LeRobot 要求 n_action_steps=1，每步都重新预测）。
- adaptive（T11a，只 ACT）：自适应重新观察规则（chunkre.adaptive），参数在 EvalConfig.adaptive；k = chunk_size（100）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml

MODELS = {
    "act": dict(repo="lerobot/act_aloha_sim_transfer_cube_human", env_type="aloha",
                env_kwargs=dict(task="AlohaTransferCube-v0"), default_k=100),
    "dp": dict(repo="lerobot/diffusion_pusht", env_type="pusht", env_kwargs=dict(), default_k=8),
}
POLICY_TYPES = {"act": "act", "dp": "diffusion"}   # policy config.json 的 "type"（本地检查点用来核对模型）
MODES = ("fixed", "te", "adaptive")
TE_COEFF = 0.01


def parse_seeds(spec) -> list[int]:
    """'0-9' / '0,1,5' / '1000-1499' / [0,1] / {'start':0,'n':10} -> list[int]（区间两端都含）。"""
    if isinstance(spec, dict):
        return list(range(int(spec["start"]), int(spec["start"]) + int(spec["n"])))
    if isinstance(spec, (list, tuple)):
        return [int(s) for s in spec]
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def k_upper_bound(model: str, pcfg) -> int:
    """从 policy config 计算 k 的上限。"""
    if model == "act":
        return int(pcfg.chunk_size)
    if model == "dp":
        return int(pcfg.horizon) - int(pcfg.n_obs_steps) + 1
    raise ValueError(f"未知模型 {model}")


def validate_k(model: str, mode: str, k: int, pcfg) -> None:
    """k 不合法时抛 ValueError。"""
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES} 之一，收到 {mode!r}")
    if mode == "te":
        if model != "act":
            raise ValueError("te（时间集成）只适用于 ACT")
        if k != 1:
            raise ValueError(f"te 模式要求 k=1（LeRobot 的时间集成要求 n_action_steps=1），收到 k={k}")
        return
    if mode == "adaptive":
        if model != "act":
            raise ValueError("adaptive（自适应重新观察）只适用于 ACT")
        if int(k) != k_upper_bound(model, pcfg):
            raise ValueError(f"adaptive 模式的计划长度 k 必须等于 chunk_size={k_upper_bound(model, pcfg)}，收到 k={k}")
        return
    ub = k_upper_bound(model, pcfg)
    if not (1 <= int(k) <= ub):
        raise ValueError(f"{model} 的执行长度 k 必须在 1..{ub} 之间，收到 k={k}")


def check_local_policy_dir(path, model: str) -> dict:
    """本地检查点目录必须有 config.json + model.safetensors，且 config 的 type 与 model 一致。返回 config.json 内容。"""
    from pathlib import Path

    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"--policy-path {p} 不是目录（应为 .../checkpoints/<步>/pretrained_model）")
    missing = [f for f in ("config.json", "model.safetensors") if not (p / f).is_file()]
    if missing:
        raise FileNotFoundError(f"--policy-path {p} 缺少 {missing}（应为 .../checkpoints/<步>/pretrained_model）")
    import json

    pc = json.loads((p / "config.json").read_text(encoding="utf-8"))
    if pc.get("type") != POLICY_TYPES[model]:
        raise ValueError(f"--policy-path {p} 的策略类型是 {pc.get('type')!r}，与 model={model!r} 不符")
    return pc


@dataclass
class EvalConfig:
    model: str                      # act / dp
    seeds: list[int]
    batch_size: int = 10            # B：同时运行的环境数
    k: int | None = None            # None = 模型默认（ACT 100 / DP 8）；te 模式下自动取 1
    mode: str = "fixed"             # fixed / te
    max_steps: int | None = None    # 只用于检查：步数上限（< 回合长度时回合被截断，记 capped=1）
    device: str = "cuda"
    allow_cpu_render: bool = False
    deterministic: bool = True      # torch 确定性开关（见 runner.set_determinism）
    name: str = ""
    extra: dict = field(default_factory=dict)
    # T3：只记录、不影响结果（不进续跑配置指纹，见 runner.NON_RESULT_FIELDS）
    log_traj: bool = False          # 逐步轨迹 -> <out>/traj/batch_*.npz（chunkre.traj）
    save_frames: list[int] = field(default_factory=list)   # 这些种子导出画面 -> <out>/frames/
    frame_every: int = 10           # 每隔几步存一张 PNG
    # T4：扰动 {kind, strength, params}（chunkre.perturb）；空 / kind=none = 无扰动（不进续跑指纹，旧结果照常续跑）
    perturb: dict = field(default_factory=dict)
    # T10a：从本地检查点加载策略（lerobot 训练输出的 .../checkpoints/<步>/pretrained_model 目录）；
    # None = hub 预训练模型 MODELS[model]["repo"]（None 时不进续跑指纹，旧结果照常续跑）
    policy_path: str | None = None
    # T11a：mode=adaptive 的参数 {m, tau, h_cmp, min_gap}（chunkre.adaptive.normalize_params）；其他模式必须为空
    # （空时不进续跑指纹，旧结果照常续跑）
    adaptive: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"model 必须是 {list(MODELS)} 之一，收到 {self.model!r}")
        if self.mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES} 之一，收到 {self.mode!r}")
        self.seeds = parse_seeds(self.seeds)
        if self.mode == "adaptive":
            from .adaptive import normalize_params

            if self.model != "act":
                raise ValueError("mode=adaptive 只支持 ACT")
            self.adaptive = normalize_params(self.adaptive)
        elif self.adaptive:
            raise ValueError(f"只有 mode=adaptive 才能给 adaptive 参数（mode={self.mode!r}）")
        else:
            self.adaptive = {}
        if self.batch_size < 1:
            raise ValueError("batch_size 必须 >= 1")
        if len(self.seeds) == 0 or len(self.seeds) % self.batch_size != 0:
            raise ValueError(f"种子数 {len(self.seeds)} 必须是 batch_size={self.batch_size} 的正整数倍")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("种子列表有重复")
        if self.policy_path in ("",):
            self.policy_path = None
        if self.policy_path is not None:
            self.policy_path = str(self.policy_path)
        self.save_frames = parse_seeds(self.save_frames) if self.save_frames not in (None, "") else []
        if int(self.frame_every) < 1:
            raise ValueError("frame_every 必须 >= 1")
        from .perturb import make_perturbation

        # 校验并补全默认参数（不建环境）：之后 self.perturb 的 params 是实际取值，续跑指纹因此与“写没写默认值”无关
        p = make_perturbation(self.perturb, self.model)
        self.perturb = dict(kind=p.kind, strength=p.strength, params=p.params)

    @property
    def perturb_active(self) -> bool:
        return self.perturb["kind"] != "none"

    @property
    def k_effective(self) -> int:
        if self.mode == "te":
            return 1
        return int(self.k) if self.k is not None else int(MODELS[self.model]["default_k"])

    def batches(self) -> list[list[int]]:
        B = self.batch_size
        return [self.seeds[i:i + B] for i in range(0, len(self.seeds), B)]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["k_effective"] = self.k_effective
        d["k_explicit"] = self.k is not None
        return d


def merge_perturb(base, kind=None, strength=None, params: dict | None = None) -> dict:
    """YAML 里的 perturb 与命令行 --perturb / --perturb-strength / --perturb-param 合并（命令行优先，params 逐键覆盖）。"""
    d = dict(base or {})
    if kind is not None:
        d["kind"] = kind
    if strength is not None:
        d["strength"] = strength
    if params:
        d["params"] = dict(d.get("params") or {}, **params)
    return d


def parse_param_kv(items) -> dict:
    """['t_lo=20', 't_hi=100'] -> {'t_lo': 20, 't_hi': 100}（值按 YAML 解析：数字 / 布尔 / 字符串）。"""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--perturb-param 需要 key=value，收到 {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def load_config(path=None, overrides: dict | None = None) -> EvalConfig:
    d = {}
    if path:
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
    overrides = dict(overrides or {})
    pk, ps, pp = (overrides.pop(x, None) for x in ("perturb_kind", "perturb_strength", "perturb_params"))
    if pk is not None or ps is not None or pp:
        d["perturb"] = merge_perturb(d.get("perturb"), pk, ps, pp)
    ap = {k: v for k, v in (overrides.pop("adaptive_params", None) or {}).items() if v is not None}
    if ap:
        d["adaptive"] = dict(d.get("adaptive") or {}, **ap)
    for key, val in overrides.items():
        if val is not None:
            d[key] = val
    known = set(EvalConfig.__dataclass_fields__)
    extra = {k: v for k, v in d.items() if k not in known}
    if extra:
        raise ValueError(f"配置里有未知字段：{sorted(extra)}")
    return EvalConfig(**d)
