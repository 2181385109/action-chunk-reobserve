"""T6b 统计函数：McNemar 精确检验、配对 bootstrap、Holm 校正、bootstrap 均值区间。全部用手算可验证的小例子。"""
import math

import numpy as np
import pytest

from chunkre.stats import (BOOT_SEED, bootstrap_mean_ci, classify, holm_adjust, mcnemar_exact, pair_counts,
                           paired_bootstrap_diff)


# ---------- McNemar 精确检验 ----------
def test_mcnemar_hand_computed():
    # b=8, c=2 -> n=10；P(X<=2) = (1 + 10 + 45)/1024 = 56/1024；双侧 p = 112/1024
    assert mcnemar_exact(8, 2) == pytest.approx(112 / 1024, abs=1e-15)
    # 对称
    assert mcnemar_exact(2, 8) == mcnemar_exact(8, 2)
    # b=5, c=0 -> n=5；P(X<=0) = 1/32；p = 2/32 = 0.0625（5 个不一致对全朝一边，也不到 0.05）
    assert mcnemar_exact(5, 0) == pytest.approx(0.0625, abs=1e-15)
    # b=6, c=0 -> p = 2/64 = 0.03125（刚好显著）
    assert mcnemar_exact(6, 0) == pytest.approx(0.03125, abs=1e-15)
    # b=1, c=0 -> n=1；2·P(X<=0) = 2·0.5 = 1
    assert mcnemar_exact(1, 0) == 1.0


def test_mcnemar_edge_cases():
    assert mcnemar_exact(0, 0) == 1.0                  # 没有不一致对
    assert mcnemar_exact(5, 5) == 1.0                  # 完全平衡：2·P(X<=5)>1，截到 1
    assert mcnemar_exact(3, 3) == 1.0                  # n=6, P(X<=3)=42/64 -> 截到 1
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024, abs=1e-15)
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 3)


def test_mcnemar_matches_scipy_binomtest_if_available():
    scipy_stats = pytest.importorskip("scipy.stats")
    for b, c in [(8, 2), (12, 30), (0, 9), (45, 60), (100, 70)]:
        assert mcnemar_exact(b, c) == pytest.approx(scipy_stats.binomtest(min(b, c), b + c, 0.5).pvalue, rel=1e-9)


def test_mcnemar_large_n_exact_and_in_unit_interval():
    p = mcnemar_exact(150, 40)
    assert 0 <= p < 1e-10      # 极端：200 个回合里 190 个不一致，几乎全朝一边
    assert mcnemar_exact(100, 100) == 1.0


def test_pair_counts():
    a = [1, 1, 1, 0, 0, 0, 1, 0]
    b = [1, 0, 0, 1, 0, 0, 1, 1]
    # 都成功：位置 0, 6 -> 2；只 A：1, 2 -> 2；只 B：3, 7 -> 2；都失败：4, 5 -> 2
    assert pair_counts(a, b) == dict(both=2, only_a=2, only_b=2, neither=2)
    with pytest.raises(ValueError):
        pair_counts([1, 0], [1])
    with pytest.raises(ValueError):
        pair_counts([1, 2], [1, 0])


# ---------- 配对 bootstrap ----------
def test_paired_bootstrap_degenerate_cases():
    # A 全成功、B 全失败：任何重抽样的差都是 1 -> 区间 [1, 1]
    est, lo, hi = paired_bootstrap_diff([1] * 20, [0] * 20)
    assert (est, lo, hi) == (1.0, 1.0, 1.0)
    # 两列相同：差恒为 0（配对的关键：相同序列的区间宽度为 0）
    x = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    assert paired_bootstrap_diff(x, x) == (0.0, 0.0, 0.0)
    # 反过来
    assert paired_bootstrap_diff([0] * 7, [1] * 7) == (-1.0, -1.0, -1.0)


def test_paired_bootstrap_point_estimate_and_two_point_example():
    # a=[1,0], b=[0,0]：点估计 0.5；重抽样差 ∈ {0, 0.5, 1}，概率 1/4, 1/2, 1/4
    # -> 2.5% 分位 = 0，97.5% 分位 = 1
    est, lo, hi = paired_bootstrap_diff([1, 0], [0, 0], n_boot=20000)
    assert est == 0.5 and lo == 0.0 and hi == 1.0


def test_paired_bootstrap_is_paired_not_independent():
    # 逐位相同的 a、b：配对区间宽度 0；若把 a、b 当独立样本重抽样，区间会很宽 —— 这里用手工独立 bootstrap 对比
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 100)
    _, lo, hi = paired_bootstrap_diff(a, a)
    assert lo == hi == 0.0
    idx_a = np.random.default_rng(1).integers(0, 100, (2000, 100))
    idx_b = np.random.default_rng(2).integers(0, 100, (2000, 100))
    indep = a[idx_a].mean(1) - a[idx_b].mean(1)
    assert np.percentile(indep, 97.5) - np.percentile(indep, 2.5) > 0.1


def test_paired_bootstrap_reproduces_manual_computation_and_is_deterministic():
    a = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=float)
    b = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0], dtype=float)
    est, lo, hi = paired_bootstrap_diff(a, b, n_boot=500)
    idx = np.random.default_rng(BOOT_SEED).integers(0, 10, size=(500, 10))
    d = a[idx].mean(1) - b[idx].mean(1)
    assert est == pytest.approx(0.3)                      # (6 - 3)/10
    assert lo == pytest.approx(np.percentile(d, 2.5)) and hi == pytest.approx(np.percentile(d, 97.5))
    assert paired_bootstrap_diff(a, b, n_boot=500) == (est, lo, hi)        # 固定随机种子：重复调用结果相同
    assert lo <= est <= hi


def test_paired_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 0], [1])
    with pytest.raises(ValueError):
        paired_bootstrap_diff([], [])


# ---------- bootstrap 均值区间 ----------
def test_bootstrap_mean_ci():
    assert bootstrap_mean_ci([3.0] * 10) == (3.0, 3.0, 3.0, 10)
    m, lo, hi, n = bootstrap_mean_ci([1, 2, 3, float("nan"), 4])      # NaN 丢掉：均值 2.5，有效 4 个
    assert m == 2.5 and n == 4 and lo <= m <= hi and 1 <= lo and hi <= 4
    m, lo, hi, n = bootstrap_mean_ci([float("nan")] * 3)
    assert n == 0 and math.isnan(m) and math.isnan(lo) and math.isnan(hi)


# ---------- Holm ----------
def test_holm_hand_computed():
    # p = (0.01, 0.04, 0.03)，m=3：排序 0.01, 0.03, 0.04 -> ×3, ×2, ×1 = 0.03, 0.06, 0.04 -> 累计最大 0.03, 0.06, 0.06
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.5, 0.5]) == pytest.approx([1.0, 1.0, 1.0])
    # 排序后 0.001·3, 0.002·2, 0.003·1 = 0.003, 0.004, 0.003 -> 累计最大 0.003, 0.004, 0.004
    assert holm_adjust([0.001, 0.002, 0.003]) == pytest.approx([0.003, 0.004, 0.004])
    assert holm_adjust([0.2]) == [0.2]
    assert holm_adjust([0.4, 0.001, 0.02]) == pytest.approx([0.4, 0.003, 0.04])


def test_holm_never_below_raw_and_monotone_in_sorted_order():
    rng = np.random.default_rng(3)
    for _ in range(50):
        p = rng.uniform(0, 1, 5)
        adj = np.array(holm_adjust(p))
        assert (adj >= p - 1e-12).all() and (adj <= 1).all()
        assert (np.diff(adj[np.argsort(p)]) >= -1e-12).all()


def test_classify():
    assert classify(0.1, 0.01) == "supported"
    assert classify(-0.1, 0.01) == "opposite"
    assert classify(0.1, 0.06) == "not_significant"
    assert classify(-0.1, 0.06) == "not_significant"
    assert classify(0.0, 0.01) == "not_significant"
