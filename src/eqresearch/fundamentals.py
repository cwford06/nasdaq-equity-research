"""Turn raw Yahoo statements into a tidy panel: one row per (ticker, fiscal year)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# field -> (statement, Yahoo row names in order of preference)
FIELDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "revenue": ("income", ("TotalRevenue", "OperatingRevenue")),
    "gross_profit": ("income", ("GrossProfit",)),
    "operating_income": ("income", ("OperatingIncome", "TotalOperatingIncomeAsReported", "EBIT")),
    "net_income": ("income", ("NetIncomeCommonStockholders", "NetIncome")),
    "ebitda": ("income", ("EBITDA", "NormalizedEBITDA")),
    "diluted_shares": ("income", ("DilutedAverageShares",)),
    "operating_cash_flow": ("cashflow", ("OperatingCashFlow",)),
    "capex": ("cashflow", ("CapitalExpenditure",)),
    "free_cash_flow": ("cashflow", ("FreeCashFlow",)),
    "total_debt": ("balance", ("TotalDebt",)),
    "cash": ("balance", ("CashCashEquivalentsAndShortTermInvestments", "CashAndCashEquivalents")),
    "equity": ("balance", ("StockholdersEquity", "CommonStockEquity")),
    "shares": ("balance", ("OrdinarySharesNumber", "ShareIssued")),
}


def _coalesce_rows(stmt: pd.DataFrame | None, names: tuple[str, ...]) -> pd.Series:
    if stmt is None or stmt.empty:
        return pd.Series(dtype=float)
    out = pd.Series(np.nan, index=stmt.columns, dtype=float)
    for name in names:
        if name in stmt.index:
            out = out.fillna(pd.to_numeric(stmt.loc[name], errors="coerce"))
    return out


def _statement_frame(raw: dict[str, pd.DataFrame], quarterly: bool = False) -> pd.DataFrame:
    cols = {}
    for field, (stmt, names) in FIELDS.items():
        key = "income_q" if quarterly else stmt
        if quarterly and stmt != "income":
            continue
        cols[field] = _coalesce_rows(raw.get(key), names)
    df = pd.DataFrame(cols)
    df.index = pd.DatetimeIndex(df.index).astype("datetime64[ns]")
    return df.sort_index()


def annual_panel(raw: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    frames = []
    for ticker, statements in raw.items():
        df = _statement_frame(statements)
        if df.empty:
            continue
        df["shares"] = df["shares"].fillna(df["diluted_shares"])
        if "free_cash_flow" in df:
            df["free_cash_flow"] = df["free_cash_flow"].fillna(df["operating_cash_flow"] + df["capex"])
        df = df[df["revenue"].notna()]
        df.insert(0, "ticker", ticker)
        frames.append(df.rename_axis("period_end").reset_index())
    if not frames:
        return pd.DataFrame(columns=["ticker", "period_end", *FIELDS])
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "period_end"], ignore_index=True)


def duplicate_share_classes(panel: pd.DataFrame) -> list[str]:
    """Tickers that report the exact same statements as another ticker (GOOG/GOOGL, BELFA/BELFB).

    Keeping both would double-count one company in the picks and in the training data,
    so all but the alphabetically first ticker are returned for removal.
    """
    latest = panel.sort_values("period_end").groupby("ticker").tail(1).sort_values("ticker").reset_index(drop=True)
    key = latest[["period_end", "revenue", "net_income"]].astype(str).agg("|".join, axis=1)
    return latest.loc[key.duplicated(keep="first"), "ticker"].tolist()


def quarterly_trends(raw: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Latest quarter vs the same quarter a year earlier (Yahoo keeps ~5 quarters).

    Only used for the live research notes: five quarters is not enough history to
    backtest, so the model itself is trained on annual statements.
    """
    rows = []
    for ticker, statements in raw.items():
        q = _statement_frame(statements, quarterly=True).dropna(subset=["revenue"])
        if len(q) < 5:
            continue
        cur, yr_ago = q.iloc[-1], q.iloc[-5]
        if not (300 <= (q.index[-1] - q.index[-5]).days <= 430) or yr_ago["revenue"] <= 0:
            continue
        op_m = cur["operating_income"] / cur["revenue"] if cur["revenue"] > 0 else np.nan
        op_m_ago = yr_ago["operating_income"] / yr_ago["revenue"]
        rows.append(
            {
                "ticker": ticker,
                "latest_quarter": q.index[-1].date(),
                "q_rev_growth_yoy": cur["revenue"] / yr_ago["revenue"] - 1,
                "q_op_margin": op_m,
                "q_d_op_margin_yoy": op_m - op_m_ago,
            }
        )
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
