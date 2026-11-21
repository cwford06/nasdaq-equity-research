import numpy as np
import pandas as pd
import pytest

from conftest import make_prices, make_statements
from eqresearch.features import build_panel, fiscal_features, forward_returns, month_end_dates


def test_growth_and_margin_math():
    panel = pd.DataFrame(
        {
            "ticker": ["A"] * 3,
            "period_end": pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31"]),
            "revenue": [100.0, 120.0, 150.0],
            "gross_profit": [40.0, 50.0, 66.0],
            "operating_income": [10.0, 15.0, 24.0],
            "net_income": [8.0, 12.0, 18.0],
            "ebitda": [12.0, 18.0, 28.0],
            "free_cash_flow": [5.0, 9.0, 15.0],
            "total_debt": [0.0] * 3, "cash": [0.0] * 3, "equity": [50.0] * 3, "shares": [10.0] * 3,
        }
    )
    f = fiscal_features(panel, report_lag_days=90).set_index("period_end")
    last = f.loc["2024-12-31"]
    assert last["rev_growth"] == pytest.approx(0.25)
    assert last["rev_growth_prev"] == pytest.approx(0.20)
    assert last["rev_accel"] == pytest.approx(0.05)
    assert last["rev_cagr_2y"] == pytest.approx(np.sqrt(1.5) - 1)
    assert last["op_margin"] == pytest.approx(0.16)
    assert last["d_op_margin"] == pytest.approx(0.16 - 0.125)
    assert last["incremental_op_margin"] == pytest.approx((24 - 15) / (150 - 120))
    assert f.loc["2024-12-31", "available_date"] == pd.Timestamp("2025-03-31")
    # first year has no prior year
    assert np.isnan(f.loc["2022-12-31", "rev_growth"])


def test_gap_in_fiscal_years_is_not_treated_as_growth():
    panel = make_statements(["A"], years=(2021, 2023))
    f = fiscal_features(panel, 90)
    assert f["rev_growth"].isna().all()


def test_forward_returns_use_first_trading_day_after_horizon():
    prices = make_prices(["A"], start="2024-01-01", end="2025-03-31")
    d = pd.Timestamp("2024-01-31")
    fwd, label_dates = forward_returns(prices, [d, pd.Timestamp("2024-12-31")], 365)
    end = label_dates[d]
    assert end >= d + pd.Timedelta(days=365) and (end - d).days <= 368
    assert fwd.loc[d, "A"] == pytest.approx(prices.loc[end, "A"] / prices.loc[d, "A"] - 1)
    assert pd.isna(label_dates[pd.Timestamp("2024-12-31")])  # horizon beyond data
    assert np.isnan(fwd.loc[pd.Timestamp("2024-12-31"), "A"])


def test_panel_is_point_in_time(settings, tickers):
    """FY2024 numbers (period end 2024-12-31) must not appear before 2025-03-31."""
    statements = make_statements(tickers)
    prices = make_prices(tickers)
    fiscal = fiscal_features(statements, settings.report_lag_days)
    panel = build_panel(prices, fiscal, settings, tickers)
    periods = panel["period_end"]
    dates = pd.Series(panel.index.get_level_values("date"), index=panel.index)
    assert ((dates - periods).dt.days >= settings.report_lag_days).all()
    feb = panel.xs(pd.Timestamp("2025-02-28"), level="date")
    assert (feb["period_end"] == pd.Timestamp("2023-12-31")).all()
    apr = panel.xs(pd.Timestamp("2025-04-30"), level="date")
    assert (apr["period_end"] == pd.Timestamp("2024-12-31")).all()


def test_valuation_uses_as_of_price(settings, tickers):
    statements = make_statements(tickers)
    prices = make_prices(tickers)
    panel = build_panel(prices, fiscal_features(statements, 90), settings, tickers)
    row = panel.iloc[0]
    date, t = panel.index[0]
    assert row["mcap"] == pytest.approx(prices.loc[date, t] * row["shares"])
    assert row["ev"] == pytest.approx(row["mcap"] + row["total_debt"] - row["cash"])
    assert row["sales_yield"] == pytest.approx(row["revenue"] / row["ev"])


def test_duplicate_share_classes_are_detected():
    from eqresearch.fundamentals import duplicate_share_classes

    a = make_statements(["GOOG"], seed=5)
    b = a.assign(ticker="GOOGL")
    c = make_statements(["MSFT"], seed=6)
    assert duplicate_share_classes(pd.concat([a, b, c])) == ["GOOGL"]


def test_month_end_dates_are_last_trading_day():
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    dates = month_end_dates(idx, "2024-01-01")
    assert [d.day for d in dates] == [31, 29, 29]  # Mar 29 2024 is a Friday
