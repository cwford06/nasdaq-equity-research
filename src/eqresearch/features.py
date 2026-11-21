"""Point-in-time feature panel.

Every row is (as-of date, ticker). A fiscal year's numbers only enter the panel
``report_lag_days`` after the fiscal year ends, and prices are sampled on the
as-of date itself, so nothing in a row was unknowable on that date (apart from
later restatements Yahoo may have applied; see the README).

The three research methods the project is built around live here:

* ``revenue_growth_features``  - growth, acceleration, 2y CAGR
* ``margin_expansion_features`` - margin levels and year-over-year change
* ``valuation_features``       - yields on EV and market cap, growth-adjusted EV/Sales
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eqresearch.config import Settings

LEVELS = (
    "revenue", "gross_profit", "operating_income", "net_income", "ebitda",
    "free_cash_flow", "total_debt", "cash", "equity", "shares",
)


def _div(a, b, positive_denominator: bool = True):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    bad = (b <= 0) if positive_denominator else (b == 0)
    return (a / b).where(~bad & b.notna())


def revenue_growth_features(f: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=f.index)
    out["rev_growth"] = _div(f["revenue"], f["revenue_l1"]) - 1
    out["rev_growth_prev"] = _div(f["revenue_l1"], f["revenue_l2"]) - 1
    out["rev_accel"] = out["rev_growth"] - out["rev_growth_prev"]
    ratio = _div(f["revenue"], f["revenue_l2"])
    out["rev_cagr_2y"] = np.sqrt(ratio) - 1
    return out


def margin_expansion_features(f: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=f.index)
    for name, num in (("gross", "gross_profit"), ("op", "operating_income"), ("net", "net_income"), ("fcf", "free_cash_flow")):
        cur = _div(f[num], f["revenue"])
        prev = _div(f[f"{num}_l1"], f["revenue_l1"])
        out[f"{name}_margin"] = cur.clip(-2, 1)
        out[f"d_{name}_margin"] = (cur - prev).clip(-1, 1)
    # Share of each new revenue dollar that reached operating income (operating leverage).
    d_rev = f["revenue"] - f["revenue_l1"]
    incremental = (f["operating_income"] - f["operating_income_l1"]) / d_rev.where(d_rev.abs() > 0.02 * f["revenue_l1"])
    out["incremental_op_margin"] = incremental.clip(-2, 2)
    return out


def valuation_features(f: pd.DataFrame) -> pd.DataFrame:
    """Needs ``price`` and the latest fundamentals on the same row."""
    out = pd.DataFrame(index=f.index)
    mcap = f["price"] * f["shares"]
    ev = mcap + f["total_debt"].fillna(0) - f["cash"].fillna(0)
    out["mcap"] = mcap
    out["log_mcap"] = np.log(mcap.where(mcap > 0))
    out["ev"] = ev
    # Yields (inverse multiples) stay meaningful when earnings are negative.
    out["sales_yield"] = _div(f["revenue"], ev)
    out["ebit_ev"] = _div(f["operating_income"], ev).clip(-1, 1)
    out["earnings_yield"] = _div(f["net_income"], mcap).clip(-1, 1)
    out["fcf_yield"] = _div(f["free_cash_flow"], mcap).clip(-1, 1)
    out["book_to_market"] = _div(f["equity"], mcap).clip(-5, 5)
    # Display multiples
    out["ev_sales"] = _div(ev, f["revenue"]).where(ev > 0)
    out["ev_ebitda"] = _div(ev, f["ebitda"]).where(ev > 0)
    out["pe"] = _div(mcap, f["net_income"])
    growth_pct = (f["rev_growth"] * 100).where(f["rev_growth"] > 0)
    out["ev_sales_to_growth"] = out["ev_sales"] / growth_pct
    return out


def fiscal_features(panel: pd.DataFrame, report_lag_days: int) -> pd.DataFrame:
    """Attach prior-year values to each fiscal row and compute growth and margin features."""
    p = panel.sort_values(["ticker", "period_end"]).reset_index(drop=True)
    g = p.groupby("ticker", sort=False)
    lagged = {}
    for k in (1, 2):
        lagged[f"period_end_l{k}"] = g["period_end"].shift(k)
        for col in ("revenue", "gross_profit", "operating_income", "net_income", "free_cash_flow"):
            lagged[f"{col}_l{k}"] = g[col].shift(k)
    p = pd.concat([p, pd.DataFrame(lagged)], axis=1)

    # Only compare consecutive fiscal years.
    ok1 = (p["period_end"] - p["period_end_l1"]).dt.days.between(300, 430)
    ok2 = ok1 & (p["period_end_l1"] - p["period_end_l2"]).dt.days.between(300, 430)
    for col in [c for c in p.columns if c.endswith("_l1")]:
        p.loc[~ok1, col] = np.nan
    for col in [c for c in p.columns if c.endswith("_l2")]:
        p.loc[~ok2, col] = np.nan

    p = pd.concat([p, revenue_growth_features(p), margin_expansion_features(p)], axis=1)
    p["available_date"] = (p["period_end"] + pd.Timedelta(days=report_lag_days)).astype("datetime64[ns]")
    keep = ["ticker", "period_end", "available_date", *LEVELS]
    keep += [c for c in p.columns if c not in keep and not c.endswith(("_l1", "_l2")) and c in FISCAL_FEATURES]
    return p[keep]


FISCAL_FEATURES = (
    "rev_growth", "rev_growth_prev", "rev_accel", "rev_cagr_2y",
    "gross_margin", "op_margin", "net_margin", "fcf_margin",
    "d_gross_margin", "d_op_margin", "d_net_margin", "d_fcf_margin", "incremental_op_margin",
)


def price_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    logret = np.log(prices).diff()
    return {
        "mom_12_1": prices.shift(21) / prices.shift(252) - 1,
        "mom_6": prices / prices.shift(126) - 1,
        "vol_1y": logret.rolling(252, min_periods=200).std() * np.sqrt(252),
        "dist_high": prices / prices.rolling(252, min_periods=200).max() - 1,
        "history_days": prices.notna().cumsum(),
    }


def month_end_dates(index: pd.DatetimeIndex, start: str) -> list[pd.Timestamp]:
    idx = index[index >= pd.Timestamp(start)]
    s = pd.Series(idx, index=idx)
    return list(s.groupby([idx.year, idx.month]).max())


def forward_returns(prices: pd.DataFrame, dates: list[pd.Timestamp], horizon_days: int) -> tuple[pd.DataFrame, pd.Series]:
    """Return from each date to the first trading day at least ``horizon_days`` later."""
    filled = prices.ffill()
    rets, label_dates = {}, {}
    last = prices.index[-1]
    for d in dates:
        target = d + pd.Timedelta(days=horizon_days)
        if target > last:
            label_dates[d] = pd.NaT
            rets[d] = pd.Series(np.nan, index=prices.columns)
            continue
        end = prices.index[prices.index.searchsorted(target)]
        label_dates[d] = end
        rets[d] = filled.loc[end] / prices.loc[d] - 1
    return pd.DataFrame(rets).T, pd.Series(label_dates)


def build_panel(prices: pd.DataFrame, fiscal: pd.DataFrame, s: Settings, tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = [t for t in (tickers or prices.columns) if t in prices.columns]
    prices = prices[tickers]
    dates = month_end_dates(prices.index, s.scan_from)
    if prices.index[-1] not in dates:
        dates.append(prices.index[-1])
    dates = sorted(set(dates))

    grid = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"]).to_frame(index=False)
    grid["date"] = grid["date"].astype("datetime64[ns]")
    grid = grid.sort_values("date")
    fis = fiscal[fiscal["ticker"].isin(tickers)].sort_values("available_date")
    df = pd.merge_asof(grid, fis, left_on="date", right_on="available_date", by="ticker", direction="backward")
    stale = (df["date"] - df["period_end"]).dt.days > s.max_staleness_days
    df.loc[stale, [c for c in df.columns if c not in ("date", "ticker")]] = np.nan

    df = df.set_index(["date", "ticker"]).sort_index()
    sampled = {name: frame.loc[dates].stack(future_stack=True) for name, frame in price_features(prices).items()}
    df["price"] = prices.loc[dates].stack(future_stack=True)
    for name, series in sampled.items():
        df[name] = series

    df = pd.concat([df, valuation_features(df)], axis=1)

    fwd, label_dates = forward_returns(prices, dates, s.horizon_days)
    df["fwd_return"] = fwd.stack(future_stack=True).reindex(df.index)
    df["label_date"] = df.index.get_level_values("date").map(label_dates)

    df["eligible"] = (
        df["rev_growth"].notna()
        & (df["mcap"] >= s.min_mcap)
        & (df["price"] >= s.min_price)
        & (df["history_days"] >= s.min_history_days)
        & df["vol_1y"].notna()
    )
    counts = df.groupby(level="date")["eligible"].sum()
    good_dates = counts[counts >= s.min_cross_section].index
    df = df[df["eligible"] & df.index.get_level_values("date").isin(good_dates)].copy()

    by_date = df.groupby(level="date")
    df["fwd_excess"] = df["fwd_return"] - by_date["fwd_return"].transform("median")
    # Log excess return is the model target: it tames the right skew of 12m returns,
    # so a handful of 10-baggers do not dominate the fit.
    log_ret = np.log1p(df["fwd_return"])
    df["fwd_log_excess"] = log_ret - log_ret.groupby(level="date").transform("median")
    return df
