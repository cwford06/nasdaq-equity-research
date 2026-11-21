"""Ensemble model for 12-month excess return, plus quantile models for the target range.

* Inputs are cross-sectional percentile ranks, so the model learns "cheaper /
  faster-growing than peers this month" rather than absolute levels that drift
  with the market.
* The point forecast averages a gradient-boosted tree model and a ridge
  regression. 10th / 90th percentile gradient-boosted models give bear / bull cases.
* The training target is the 12-month *log* return in excess of the median
  eligible stock (log tames the right skew of single-stock returns). The price
  target adds an assumed market return:  P * (1 + market_return) * exp(excess).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from eqresearch.screens import SCORE_COLS

MODEL_FEATURES: tuple[str, ...] = (
    # revenue growth
    "rev_growth", "rev_growth_prev", "rev_accel", "rev_cagr_2y",
    # margins and margin expansion
    "gross_margin", "op_margin", "fcf_margin",
    "d_gross_margin", "d_op_margin", "d_net_margin", "d_fcf_margin", "incremental_op_margin",
    # valuation
    "sales_yield", "ebit_ev", "earnings_yield", "fcf_yield", "book_to_market",
    # the one price signal with decades of out-of-sample evidence
    "mom_12_1",
    # pillar scores
    *SCORE_COLS,
)

# Size, volatility and short-term price features. With ~2 years of point-in-time history
# the first version of the model used these mostly to bet on the recent market regime
# (small, volatile semis), which then reversed. They stay in the panel for the signal
# study and for the "v1" comparison in the backtest report, but not in the live model.
RISK_FEATURES: tuple[str, ...] = ("mom_6", "vol_1y", "dist_high", "log_mcap")
V1_FEATURES: tuple[str, ...] = MODEL_FEATURES + RISK_FEATURES

# Tree settings. "strong" (the default) is shallower and more regularised, because a few
# hundred stocks x ~18 overlapping months is very little data for a flexible model.
TREE_PARAMS = {
    "standard": dict(max_iter=250, learning_rate=0.04, max_depth=3, leaf_div=60, min_leaf=20, l2_regularization=1.0),
    "strong": dict(max_iter=120, learning_rate=0.03, max_depth=2, leaf_div=30, min_leaf=50, l2_regularization=5.0),
}


def rank_features(df: pd.DataFrame, cols=MODEL_FEATURES) -> pd.DataFrame:
    return df[list(cols)].groupby(level="date").rank(pct=True)


def _date_weights(index: pd.MultiIndex) -> np.ndarray:
    dates = index.get_level_values("date")
    counts = pd.Series(dates).map(pd.Series(dates).value_counts())
    return (1.0 / counts).to_numpy()


@dataclass
class ExcessReturnModel:
    seed: int = 0
    quantiles: tuple[float, float] = (0.1, 0.9)
    features: tuple[str, ...] = MODEL_FEATURES
    regularization: str = "strong"
    models: dict = field(default_factory=dict)
    clip: tuple[float, float] = (np.nan, np.nan)

    def _gbr(self, n: int, **kw) -> HistGradientBoostingRegressor:
        p = dict(TREE_PARAMS[self.regularization])
        leaf = max(p.pop("min_leaf"), n // p.pop("leaf_div"))
        return HistGradientBoostingRegressor(min_samples_leaf=leaf, random_state=self.seed, **p, **kw)

    def fit(self, df: pd.DataFrame, target: str = "fwd_log_excess") -> "ExcessReturnModel":
        train = df[df[target].notna()]
        X = rank_features(train, self.features)
        # Early windows can have features with no variation (e.g. 3-year growth before
        # three statements exist); trees cannot bin those, and they carry no signal.
        self.features = tuple(c for c in self.features if X[c].nunique(dropna=True) >= 2)
        X = X[list(self.features)]
        y = train[target]
        lo, hi = y.quantile([0.02, 0.98])
        self.clip = (float(lo), float(hi))
        y = y.clip(lo, hi)
        w = _date_weights(train.index)
        n = len(train)
        alpha = 10.0 if self.regularization == "standard" else 50.0
        self.models = {
            "gbr": self._gbr(n).fit(X, y, sample_weight=w),
            "ridge": Ridge(alpha=alpha).fit(X.fillna(0.5) - 0.5, y, sample_weight=w),
            "q_lo": self._gbr(n, loss="quantile", quantile=self.quantiles[0]).fit(X, y, sample_weight=w),
            "q_hi": self._gbr(n, loss="quantile", quantile=self.quantiles[1]).fit(X, y, sample_weight=w),
        }
        self._train = (X, y, w)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = rank_features(df, self.features)
        point = self._point(X)
        lo = self.models["q_lo"].predict(X)
        hi = self.models["q_hi"].predict(X)
        return pd.DataFrame(
            {"pred_excess": point, "pred_lo": np.minimum(lo, point), "pred_hi": np.maximum(hi, point)},
            index=df.index,
        )

    def importance(self, n_repeats: int = 5) -> pd.Series:
        """Permutation importance of the point model on its own training data."""
        X, y, w = self._train
        rng = np.random.default_rng(self.seed)
        base = r2_score(y, self._point(X), sample_weight=w)
        drops = {}
        for col in X.columns:
            scores = []
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[col] = rng.permutation(Xp[col].to_numpy())
                scores.append(r2_score(y, self._point(Xp), sample_weight=w))
            drops[col] = base - float(np.mean(scores))
        return pd.Series(drops).sort_values(ascending=False)

    def _point(self, X: pd.DataFrame) -> np.ndarray:
        return 0.5 * (self.models["gbr"].predict(X) + self.models["ridge"].predict(X.fillna(0.5) - 0.5))


def price_targets(price: pd.Series, preds: pd.DataFrame, market_return: float) -> pd.DataFrame:
    """Predictions are 12m log excess returns vs the median stock.

    target = price * (1 + market_return) * exp(predicted log excess)
    """

    def tgt(log_excess):
        return price * (1 + market_return) * np.exp(log_excess)

    target = tgt(preds["pred_excess"])
    return pd.DataFrame(
        {
            "target": target,
            "bear": tgt(preds["pred_lo"]),
            "bull": tgt(preds["pred_hi"]),
            "implied_return": target / price - 1,
        },
        index=preds.index,
    )
