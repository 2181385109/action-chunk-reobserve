"""T6b 统计函数（纯计算，不依赖仿真；单元测试用手算的小例子验证）。严格按 PREREGISTRATION.md 实现。

- mcnemar_exact：按种子配对的双侧精确 McNemar 检验（只看不一致对，二项分布 p=0.5 的精确检验）。
- paired_bootstrap_diff：配对 bootstrap（按种子重抽样）给出成功率差的 95% 百分位区间。
- bootstrap_mean_ci：单格某指标均值的 bootstrap 95% 百分位区间（按回合重抽样，忽略 NaN）。
- holm_adjust：Holm 逐步校正（单调化的校正 p 值）。
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

N_BOOT = 10000
BOOT_SEED = 20261007          # 事先登记的 bootstrap 随机种子（每次调用都用它新建一个生成器，结果与调用顺序无关）
ALPHA = 0.05


def pair_counts(a, b) -> dict:
    """a、b：按同一批种子对齐的 0/1 序列（A 策略、B 策略）。返回 2x2 配对计数。

    both = 都成功，only_a = 只有 A 成功（b_ 不一致对），only_b = 只有 B 成功（c_ 不一致对），neither = 都失败。
    """
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"配对序列长度必须相同：{a.shape} vs {b.shape}")
    if not (np.isin(a, (0, 1)).all() and np.isin(b, (0, 1)).all()):
        raise ValueError("成功必须是 0/1")
    return dict(both=int(((a == 1) & (b == 1)).sum()), only_a=int(((a == 1) & (b == 0)).sum()),
                only_b=int(((a == 0) & (b == 1)).sum()), neither=int(((a == 0) & (b == 0)).sum()))


def mcnemar_exact(n_only_a: int, n_only_b: int) -> float:
    """双侧精确 McNemar p 值：n = b+c 个不一致对，H0 下“只有 A 成功”的个数 ~ Binomial(n, 1/2)。

    p = min(1, 2 · P(X <= min(b, c)))。n = 0 时 p = 1。用整数 / 分数精确计算，最后转 float。
    例：b=8, c=2 -> n=10, P(X<=2) = (1+10+45)/1024 = 56/1024，p = 112/1024 = 0.109375。
    """
    b, c = int(n_only_a), int(n_only_b)
    if b < 0 or c < 0:
        raise ValueError("计数不能为负")
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    tail = sum(math.comb(n, i) for i in range(m + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2 ** n)))


def paired_bootstrap_diff(a, b, n_boot: int = N_BOOT, seed: int = BOOT_SEED, alpha: float = ALPHA):
    """成功率差 mean(a) - mean(b) 的配对 bootstrap（对种子下标有放回重抽样，A、B 用同一组下标）。

    返回 (点估计, 下界, 上界)；区间 = 重抽样差值的 alpha/2 与 1-alpha/2 百分位（numpy 默认线性插值）。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("配对序列必须等长且非空")
    n = a.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def bootstrap_mean_ci(x, n_boot: int = N_BOOT, seed: int = BOOT_SEED, alpha: float = ALPHA):
    """某指标均值的 bootstrap 百分位区间（对回合有放回重抽样）。NaN 先丢掉。

    返回 (均值, 下界, 上界, 有效回合数)；没有有效回合时全是 nan、有效数 0。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    m = x[idx].mean(axis=1)
    lo, hi = np.percentile(m, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi), int(n)


def holm_adjust(pvals) -> list[float]:
    """Holm 校正：p 从小到大排序，第 i 小（i 从 1 数）乘 (m - i + 1)，再取累计最大值（保证单调），上限 1。

    返回与输入同顺序的校正 p 值。校正 p < α 等价于 Holm 逐步程序拒绝该假设。
    例：p = (0.01, 0.04, 0.03)，m=3 -> 排序 0.01, 0.03, 0.04 -> 0.03, 0.06, 0.06 -> 原顺序 (0.03, 0.06, 0.06)。
    """
    p = [float(x) for x in pvals]
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def classify(diff: float, p_holm: float, alpha: float = ALPHA) -> str:
    """假设（“A 的成功率高于 B”）的判定：
    supported = 差 > 0 且 Holm 校正 p < alpha；opposite = 差 < 0 且 Holm 校正 p < alpha（“相反结果”）；否则 not_significant。
    """
    if p_holm < alpha:
        if diff > 0:
            return "supported"
        if diff < 0:
            return "opposite"
    return "not_significant"


def gap_change(short_c, long_c, short_0, long_0, n_boot: int = N_BOOT, seed: int = BOOT_SEED, alpha: float = ALPHA) -> dict:
    """“差距变化”（探索性，不检验）：短 k 与长 k 的成败差，在扰动下比无扰动时大多少。

    四个 0/1 序列按同一批种子对齐。每个种子 i：
        gap_change_i = (short_c[i] - long_c[i]) - (short_0[i] - long_0[i])
    取平均，给对种子有放回重抽样的 bootstrap 百分位区间（四个格子用同一组重抽样下标；
    等价于对 gap_change_i 做 bootstrap_mean_ci，随机种子同为 BOOT_SEED）。
    恒等式：mean(gap_change_i) = (rate_short_c - rate_long_c) - (rate_short_0 - rate_long_0)。
    正值 = 短 k 在扰动下相对少吃亏（或多受益）；负值相反。
    """
    arrs = [np.asarray(x) for x in (short_c, long_c, short_0, long_0)]
    if any(a.shape != arrs[0].shape or a.ndim != 1 or a.size == 0 for a in arrs):
        raise ValueError("四个序列必须等长、非空")
    if not all(np.isin(a, (0, 1)).all() for a in arrs):
        raise ValueError("成功必须是 0/1")
    sc, lc, s0, l0 = (a.astype(np.float64) for a in arrs)
    d = (sc - lc) - (s0 - l0)
    mean, lo, hi, n = bootstrap_mean_ci(d, n_boot=n_boot, seed=seed, alpha=alpha)
    return dict(n=n, gap_pert=float(sc.mean() - lc.mean()), gap_none=float(s0.mean() - l0.mean()),
                gap_change=mean, ci_lo=lo, ci_hi=hi,
                rate_short_pert=float(sc.mean()), rate_long_pert=float(lc.mean()),
                rate_short_none=float(s0.mean()), rate_long_none=float(l0.mean()))
