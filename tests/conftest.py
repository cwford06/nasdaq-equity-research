import numpy as np
import pandas as pd
import pytest

from eqresearch.config import Settings


def make_statements(tickers, years=(2021, 2022, 2023, 2024, 2025), seed=0):
    """Annual panel with Dec fiscal years and random but sane numbers."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in tickers:
        rev = rng.uniform(1e9, 5e9)
        growth = rng.uniform(-0.1, 0.5)
        margin = rng.uniform(0.05, 0.3)
        for y in years:
            rev *= 1 + growth + rng.normal(0, 0.05)
            margin += rng.normal(0, 0.02)
            rows.append(
                {
                    "ticker": t,
                    "period_end": pd.Timestamp(f"{y}-12-31"),
                    "revenue": rev,
                    "gross_profit": rev * (margin + 0.3),
                    "operating_income": rev * margin,
                    "net_income": rev * margin * 0.8,
                    "ebitda": rev * (margin + 0.05),
                    "free_cash_flow": rev * margin * 0.7,
                    "total_debt": rev * 0.2,
                    "cash": rev * 0.1,
                    "equity": rev * 0.5,
                    "shares": 1e8,
                }
            )
    return pd.DataFrame(rows)


def make_prices(tickers, start="2022-01-03", end="2026-06-30", seed=1, drift=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    data = {}
    for t in tickers:
        mu = (drift or {}).get(t, 0.0003)
        data[t] = 50 * np.exp(np.cumsum(rng.normal(mu, 0.02, len(idx))))
    return pd.DataFrame(data, index=idx.astype("datetime64[ns]"))


@pytest.fixture
def settings():
    return Settings(min_mcap=0, min_price=0, min_cross_section=5, min_train_dates=3, top_n=5,
                    scan_from="2023-01-31", price_start="2022-01-03")


@pytest.fixture
def tickers():
    return [f"T{i:02d}" for i in range(40)]
