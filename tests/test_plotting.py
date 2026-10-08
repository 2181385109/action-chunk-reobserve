"""Plot helper: error-bar lengths must be >= 0 even when a Wilson bound differs from the estimate by float rounding."""
import numpy as np
import pytest

from chunkre.metrics import wilson_ci
from chunkre.plotting import yerr_from_ci


def test_wilson_lower_bound_of_zero_successes_is_tiny_positive():
    lo, hi = wilson_ci(0, 200)           # the real k20__delay10 case (0/200)
    assert 0 < lo < 1e-15 and 0.0 - lo < 0   # estimate 0.0 minus lo is a tiny negative number -> raw yerr invalid


def test_yerr_clipped_to_nonnegative_and_values_unchanged():
    lo, hi = wilson_ci(0, 200)
    y = yerr_from_ci([0.0, 0.5], [lo, 0.4], [hi, 0.6])
    assert y.shape == (2, 2) and (y >= 0).all()
    assert y[0, 0] == 0.0 and y[1, 0] == pytest.approx(hi)
    assert y[0, 1] == pytest.approx(0.1) and y[1, 1] == pytest.approx(0.1)
    # all-successes mirror case: upper bound slightly below 1.0 estimate
    lo1, hi1 = wilson_ci(200, 200)
    y1 = yerr_from_ci([1.0], [lo1], [hi1])
    assert (y1 >= 0).all()


def test_matplotlib_errorbar_accepts_clipped_yerr_and_rejects_raw():
    plt = pytest.importorskip("matplotlib.pyplot")
    import matplotlib
    matplotlib.use("Agg")
    lo, hi = wilson_ci(0, 200)
    fig, ax = plt.subplots()
    ax.errorbar([0, 1], [0.0, 0.0], yerr=yerr_from_ci([0.0, 0.0], [lo, lo], [hi, hi]))   # must not raise
    with pytest.raises(ValueError, match="yerr"):
        ax.errorbar([0, 1], [0.0, 0.0], yerr=[np.array([0.0, 0.0]) - lo, np.array([hi, hi])])
    plt.close(fig)
