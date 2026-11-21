import numpy as np
import pandas as pd
import pytest

from conftest import make_prices, make_statements
from eqresearch import backtest as bt
from eqresearch.features import build_panel, fiscal_features
from eqresearch.model import ExcessReturnModel, price_targets
from eqresearch.screens import PILLARS, add_pillar_scores


def _panel(settings, tickers, drift=None):
    statements = make_statements(tickers)
    prices = make_prices(tickers, drift=drift)
    panel = add_pillar_scores(build_panel(prices, fiscal_features(statements, 90), settings, tickers))
    return panel, prices


def test_pillar_scores_are_percentiles(settings, tickers):
    panel, _ = _panel(settings, tickers)
    for p in PILLARS:
        s = panel[f"{p}_score"].dropna()
        assert s.between(0, 1).all()
    assert panel["screens_passed"].between(0, 3).all()


def test_walk_forward_never_trains_on_unrealised_labels(settings, tickers, monkeypatch):
    panel, _ = _panel(settings, tickers)
    seen = []
    real_fit = ExcessReturnModel.fit

    def spy(self, df, target="fwd_log_excess"):
        seen.append(df["label_date"].max())
        return real_fit(self, df, target)

    monkeypatch.setattr(ExcessReturnModel, "fit", spy)
    preds = bt.walk_forward(panel, settings)
    formation = sorted(preds.index.get_level_values("date").unique())
    assert len(seen) == len(formation) > 0
    for t, max_label in zip(formation, seen):
        assert max_label <= t


def test_model_recovers_planted_signal(settings):
    """If 12m excess return is driven by revenue growth, the model should rank on it out of sample."""
    rng = np.random.default_rng(3)
    rows = []
    dates = pd.date_range("2023-01-31", periods=24, freq="ME")
    for d in dates:
        for i in range(150):
            g = rng.normal(0.1, 0.2)
            rows.append({"date": d, "ticker": f"S{i}", "rev_growth": g,
                         "fwd_log_excess": 0.8 * g + rng.normal(0, 0.1)})
    df = pd.DataFrame(rows).set_index(["date", "ticker"])
    train, test = df.loc[dates[:16]], df.loc[dates[20:]]
    m = ExcessReturnModel(features=("rev_growth",)).fit(train)
    pred = m.predict(test)["pred_excess"]
    ic = pred.rank().corr(test["fwd_log_excess"].rank())
    assert ic > 0.5


def test_price_targets_are_ordered():
    idx = pd.Index(["A", "B", "C"])
    preds = pd.DataFrame({"pred_excess": [0.1, -0.2, 0.0], "pred_lo": [-0.3, -0.5, -0.1], "pred_hi": [0.4, 0.1, 0.2]}, index=idx)
    price = pd.Series([100.0, 50.0, 10.0], index=idx)
    t = price_targets(price, preds, market_return=0.08)
    assert (t["bear"] <= t["target"]).all() and (t["target"] <= t["bull"]).all()
    assert t.loc["C", "target"] == pytest.approx(10.8)
    assert t.loc["A", "implied_return"] == pytest.approx(1.08 * np.exp(0.1) - 1)


def test_selection_respects_screens(settings, tickers):
    panel, _ = _panel(settings, tickers)
    cs = panel.xs(panel.index.get_level_values("date").max(), level="date").copy()
    cs["pred_excess"] = np.linspace(1, 0, len(cs))
    sel = bt.select(cs, settings)
    assert len(sel["ml_screened"]) <= settings.top_n
    assert (cs.loc[sel["ml_screened"], "screens_passed"] >= settings.min_screens_passed).all()
    assert sel["ml_only"] == cs.nlargest(settings.top_n, "pred_excess").index.tolist()


def test_sector_cap_limits_concentration():
    df = pd.DataFrame(
        {"score": [9, 8, 7, 6, 5, 4], "sector": ["Tech", "Tech", "Tech", "Health", "Energy", "Tech"]},
        index=list("ABCDEF"),
    )
    assert bt.top_n(df, "score", 4, max_per_sector=2) == ["A", "B", "D", "E"]
    assert bt.top_n(df, "score", 4, max_per_sector=None) == ["A", "B", "C", "D"]


def test_cohort_return_is_buy_and_hold_equal_weight():
    idx = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"A": [10, 11, 12, 13.0], "B": [20, 20, 18, 22.0]}, index=idx)
    r = bt.cohort_daily(prices, ["A", "B"], idx[0], idx[-1])
    wealth = (1 + r).prod()
    assert wealth == pytest.approx(0.5 * (13 / 10) + 0.5 * (22 / 20))


def test_full_backtest_runs(settings, tickers):
    panel, prices = _panel(settings, tickers)
    res = bt.run_backtest(panel, prices, settings)
    assert set(bt.STRATEGIES) <= set(res.daily.columns)
    assert not res.cohorts.empty and not res.factor_ic.empty
    assert res.ic["ml_ic"].between(-1, 1).all()
