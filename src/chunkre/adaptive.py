"""T11a 自适应重新观察规则（mode=adaptive，只支持 ACT）。纯 numpy，不依赖仿真 / torch，单元测试覆盖。

规则（参数 m、tau、h_cmp=20、min_gap=20；动作块长 L = ACT 的 chunk_size = 100）：
- 默认像 k=100 一样执行当前计划（动作块）；
- 自上次换计划起每过 m 步（j = m, 2m, …，j < L；j = 自上次换计划起已执行的步数 = 本步要执行的旧计划下标），
  用当前观测调用一次策略得到“候选块”（记一次策略调用），但不执行。差异 =
  候选块前 h_cmp 个动作与旧计划接下来 h_cmp 个动作（下标 j … j+h_cmp−1）逐步 L2 距离的平均值；
  旧计划剩余不足 h_cmp 步时只比较重叠部分（overlap = min(h_cmp, L − j)）。
  动作 = 环境动作向量（aloha 14 维：两臂各 6 个关节目标角度，弧度 + 1 个夹爪开合度，归一化 0..1），整个向量算 L2；
- 差异 > tau 且 j >= min_gap：换成候选块，本步就执行它的第 1 个动作（下标 0），j 归零。否则继续执行旧计划；
- 旧计划执行完（j = L）：正常重新预测并换成新块（“块用完”），也算换计划。第 0 步的第一次预测是“初始计划”，不算换计划。

批量：B 个环境一起跑（runner.run_batch），任一环境本步需要预测时对整批调用一次 predict_action_chunk，
只有需要的环境用它的那一行。每个环境各自记自己的调用次数 / 换计划 / 比较。
已结束的环境（gymnasium 自动 reset 后还会被步进，结果被丢掉）照常按旧计划执行、块用完时换块，
但不做比较、不计数、不记录——与 fixed 模式里已结束环境的处理一致，不影响任何结果。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# 逐回合 CSV 里 adaptive 模式额外的列（只有 adaptive 模式的 CSV 有，排在最后）
ADAPTIVE_COLUMNS = ["n_checks", "n_replans_diff", "n_replans_exhaust"]
ADAPTIVE_DEFAULTS = dict(h_cmp=20, min_gap=20)
ADAPTIVE_KEYS = ("m", "tau", "h_cmp", "min_gap")
# 换计划原因编码（npz 的 rp_cause）
CAUSE_INITIAL, CAUSE_EXHAUST, CAUSE_DIFF = 0, 1, 2


def parse_tau(x) -> float:
    """tau：数字，或 'inf' / 'infinity' / '.inf'（YAML 写法）= 永远不触发。"""
    if isinstance(x, str):
        s = x.strip().lower().lstrip("+")
        if s in ("inf", "infinity", ".inf"):
            return math.inf
        x = float(s)
    v = float(x)
    if math.isnan(v) or v < 0:
        raise ValueError(f"adaptive.tau 必须 >= 0（或 inf），收到 {x!r}")
    return v


def normalize_params(d: dict | None) -> dict:
    """校验并补全 adaptive 参数，返回 {m, tau, h_cmp, min_gap}（m、tau 必须给）。"""
    d = dict(d or {})
    unknown = set(d) - set(ADAPTIVE_KEYS)
    if unknown:
        raise ValueError(f"adaptive 参数有未知字段 {sorted(unknown)}（只认 {list(ADAPTIVE_KEYS)}）")
    missing = [k for k in ("m", "tau") if d.get(k) is None]
    if missing:
        raise ValueError(f"mode=adaptive 必须给 adaptive.{missing}（--adapt-m / --adapt-tau）")
    def opt(key):
        return int(ADAPTIVE_DEFAULTS[key] if d.get(key) is None else d[key])

    out = dict(m=int(d["m"]), tau=parse_tau(d["tau"]), h_cmp=opt("h_cmp"), min_gap=opt("min_gap"))
    if out["m"] < 1:
        raise ValueError(f"adaptive.m 必须 >= 1，收到 {out['m']}")
    if out["h_cmp"] < 1:
        raise ValueError(f"adaptive.h_cmp 必须 >= 1，收到 {out['h_cmp']}")
    if out["min_gap"] < 0:
        raise ValueError(f"adaptive.min_gap 必须 >= 0，收到 {out['min_gap']}")
    return out


def plan_diff(old_plan: np.ndarray, j: int, cand: np.ndarray, h_cmp: int) -> tuple[float, int]:
    """差异 = mean_{s<overlap} ||cand[s] − old_plan[j+s]||_2，overlap = min(h_cmp, len(old_plan) − j, len(cand))。
    返回 (差异, overlap)。overlap = 0（旧计划已用完）时差异为 nan。float64 计算。"""
    old_plan = np.asarray(old_plan, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    ov = int(max(0, min(int(h_cmp), old_plan.shape[0] - int(j), cand.shape[0])))
    if ov == 0:
        return float("nan"), 0
    d = np.linalg.norm(cand[:ov] - old_plan[j:j + ov], axis=1)
    return float(d.mean()), ov


class AdaptiveController:
    """一批 B 个环境的自适应执行状态。每个环境步的用法（runner.run_batch）：

        need = ctrl.need_predict(alive)          # (B,) 本步哪些环境需要策略输出
        chunks = predict(obs) if need.any() else None   # (B, L, A) numpy：整批调用一次
        action = ctrl.step(t, alive, chunks)     # (B, A)：本步执行的动作；ctrl.called[i] = 本步环境 i 是否记一次调用
    """

    def __init__(self, n_envs: int, params: dict, chunk_len: int):
        self.p = normalize_params(params)
        self.B = int(n_envs)
        self.L = int(chunk_len)
        self.plans: list[np.ndarray | None] = [None] * self.B
        self.j = np.zeros(self.B, dtype=np.int64)        # 本步要执行的计划下标 = 自上次换计划起已执行的步数
        self.n_calls = np.zeros(self.B, dtype=np.int64)
        self.n_checks = np.zeros(self.B, dtype=np.int64)
        self.n_diff = np.zeros(self.B, dtype=np.int64)
        self.n_exhaust = np.zeros(self.B, dtype=np.int64)
        self.called = np.zeros(self.B, dtype=bool)
        self.checks: list[tuple] = []     # (env, t, j, diff, overlap, eligible, switched)
        self.replans: list[tuple] = []    # (env, t, cause)
        self.plan_j: list[np.ndarray] = []   # 每步：各环境执行的计划下标（已结束 = -1）

    def _is_check(self, i: int) -> bool:
        j = int(self.j[i])
        return 0 < j < self.L and j % self.p["m"] == 0

    def need_predict(self, alive: np.ndarray) -> np.ndarray:
        alive = np.asarray(alive, dtype=bool)
        need = np.zeros(self.B, dtype=bool)
        for i in range(self.B):
            if self.plans[i] is None or self.j[i] >= self.L:
                need[i] = True
            elif alive[i] and self._is_check(i):
                need[i] = True
        return need

    def step(self, t: int, alive: np.ndarray, chunks: np.ndarray | None) -> np.ndarray:
        alive = np.asarray(alive, dtype=bool)
        need = self.need_predict(alive)
        if need.any() and chunks is None:
            raise ValueError(f"第 {t} 步有环境需要策略输出，但 chunks=None")
        self.called[:] = False
        acts = []
        p = self.p
        for i in range(self.B):
            if self.plans[i] is None:                       # 第 0 步：初始计划
                self.plans[i], self.j[i] = np.asarray(chunks[i]), 0
                self._replan_event(i, t, CAUSE_INITIAL, alive[i])
            elif self.j[i] >= self.L:                       # 旧计划执行完：块用完
                self.plans[i], self.j[i] = np.asarray(chunks[i]), 0
                self._replan_event(i, t, CAUSE_EXHAUST, alive[i])
            elif alive[i] and self._is_check(i):            # 每 m 步比较一次
                j = int(self.j[i])
                diff, ov = plan_diff(self.plans[i], j, chunks[i], p["h_cmp"])
                eligible = j >= p["min_gap"]
                switch = bool(eligible and diff > p["tau"])
                self.n_checks[i] += 1
                self.n_calls[i] += 1
                self.called[i] = True
                self.checks.append((i, int(t), j, diff, ov, bool(eligible), switch))
                if switch:
                    self.plans[i], self.j[i] = np.asarray(chunks[i]), 0
                    self.n_diff[i] += 1
                    self.replans.append((i, int(t), CAUSE_DIFF))
            acts.append(self.plans[i][self.j[i]])
        self.plan_j.append(np.where(alive, self.j, -1).astype(np.int16))
        self.j += 1
        return np.stack(acts)

    def _replan_event(self, i: int, t: int, cause: int, alive_i: bool) -> None:
        if not alive_i:
            return
        self.n_calls[i] += 1
        self.called[i] = True
        if cause == CAUSE_EXHAUST:
            self.n_exhaust[i] += 1
        self.replans.append((i, int(t), cause))

    def episode_info(self, i: int) -> dict:
        return dict(n_checks=int(self.n_checks[i]), n_replans_diff=int(self.n_diff[i]),
                    n_replans_exhaust=int(self.n_exhaust[i]))

    def arrays(self, seeds, steps) -> dict:
        """一批的记录（写进 <out>/adaptive/batch_<批号>.npz）。"""
        ck = self.checks
        rp = self.replans
        return dict(
            seeds=np.asarray(seeds, dtype=np.int64), steps=np.asarray(steps, dtype=np.int64),
            m=np.int64(self.p["m"]), tau=np.float64(self.p["tau"]), h_cmp=np.int64(self.p["h_cmp"]),
            min_gap=np.int64(self.p["min_gap"]), chunk_len=np.int64(self.L),
            chk_env=np.array([c[0] for c in ck], dtype=np.int16), chk_t=np.array([c[1] for c in ck], dtype=np.int16),
            chk_j=np.array([c[2] for c in ck], dtype=np.int16), chk_diff=np.array([c[3] for c in ck], dtype=np.float64),
            chk_overlap=np.array([c[4] for c in ck], dtype=np.int16),
            chk_eligible=np.array([c[5] for c in ck], dtype=bool), chk_switch=np.array([c[6] for c in ck], dtype=bool),
            rp_env=np.array([r[0] for r in rp], dtype=np.int16), rp_t=np.array([r[1] for r in rp], dtype=np.int16),
            rp_cause=np.array([r[2] for r in rp], dtype=np.int8),
            plan_j=np.stack(self.plan_j) if self.plan_j else np.zeros((0, self.B), dtype=np.int16),
        )

    def write_npz(self, path: Path, seeds, steps) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays(seeds, steps))


def load_checks(run_dir: Path) -> list[dict]:
    """读一个运行目录下所有 adaptive/batch_*.npz 的比较记录，返回逐条 dict（seed, t, j, diff, overlap, eligible, switched）。"""
    out = []
    for p in sorted((Path(run_dir) / "adaptive").glob("batch_*.npz")):
        z = np.load(p)
        seeds = z["seeds"]
        for e, t, j, d, ov, el, sw in zip(z["chk_env"], z["chk_t"], z["chk_j"], z["chk_diff"], z["chk_overlap"],
                                          z["chk_eligible"], z["chk_switch"]):
            out.append(dict(seed=int(seeds[e]), t=int(t), j=int(j), diff=float(d), overlap=int(ov),
                            eligible=bool(el), switched=bool(sw)))
    return out


__all__ = ["ADAPTIVE_COLUMNS", "ADAPTIVE_DEFAULTS", "CAUSE_INITIAL", "CAUSE_EXHAUST", "CAUSE_DIFF", "parse_tau",
           "normalize_params", "plan_diff", "AdaptiveController", "load_checks"]
