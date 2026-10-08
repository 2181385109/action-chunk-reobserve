"""Plot helpers (no statistics): shared by analyze_main.py and make_main_figures.py."""
from __future__ import annotations

import numpy as np


def yerr_from_ci(y, lo, hi) -> np.ndarray:
    """Matplotlib errorbar `yerr` (2 x N) from a point estimate and interval bounds, lengths clipped to >= 0.

    Interval bounds can differ from the point estimate by float rounding (e.g. Wilson lower bound of
    1.7e-18 for 0/200), which would give a tiny negative length and make matplotlib raise
    "'yerr' must not contain negative values". Only the drawn length is clipped; the CSV/JSON values are untouched.
    """
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    return np.vstack([np.clip(y - lo, 0.0, None), np.clip(hi - y, 0.0, None)])


# Same palette/markers as the exploratory figures in analyze_main.py (Okabe-Ito; marker + line style give a second encoding).
POLICY_ORDER = ["k20", "k50", "k100", "te"]
POLICY_LABEL = {"k20": "k=20", "k50": "k=50", "k100": "k=100", "te": "temporal ensembling"}
POLICY_STYLE = {"k20": ("#009E73", "s", "-"), "k50": ("#E69F00", "^", "-"), "k100": ("#D55E00", "D", "-"),
                "te": ("#000000", "v", "--")}
