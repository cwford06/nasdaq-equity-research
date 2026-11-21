"""Rule-based pillar scores: revenue growth, margin expansion, valuation.

Each pillar is the average cross-sectional percentile rank of its metrics on a
given date, so a score of 0.8 means "better than 80% of eligible stocks that
month". A stock *passes* a pillar screen when its score is at least 0.5.
"""

from __future__ import annotations

import pandas as pd

PILLARS: dict[str, tuple[str, ...]] = {
    "growth": ("rev_growth", "rev_accel", "rev_cagr_2y"),
    "margin": ("d_gross_margin", "d_op_margin", "d_fcf_margin", "incremental_op_margin"),
    "value": ("sales_yield", "ebit_ev", "fcf_yield", "earnings_yield"),
}
SCORE_COLS = tuple(f"{p}_score" for p in PILLARS) + ("composite_score",)


def cross_sectional_rank(df: pd.DataFrame, cols) -> pd.DataFrame:
    return df[list(cols)].groupby(level="date").rank(pct=True)


def add_pillar_scores(df: pd.DataFrame, pass_threshold: float = 0.5) -> pd.DataFrame:
    out = df.copy()
    for pillar, cols in PILLARS.items():
        out[f"{pillar}_score"] = cross_sectional_rank(out, cols).mean(axis=1, skipna=True)
    out["composite_score"] = out[[f"{p}_score" for p in PILLARS]].mean(axis=1, skipna=True)
    passed = pd.concat([(out[f"{p}_score"] >= pass_threshold).rename(p) for p in PILLARS], axis=1)
    for p in PILLARS:
        out[f"pass_{p}"] = passed[p]
    out["screens_passed"] = passed.sum(axis=1)
    return out
