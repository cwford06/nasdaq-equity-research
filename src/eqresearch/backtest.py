"""Walk-forward backtest.

On every month-end as-of date ``t`` the model is refit using only rows whose
12-month label was fully realised on or before ``t`` (the label window ended
before the decision date, so nothing overlaps the future). It then ranks the
cross-section at ``t`` and sets price targets.

Each month's picks form a cohort held for 12 months. As in Jegadeesh-Titman
studies, the strategy's daily return is the average daily return of all cohorts
alive that day, so every month gets equal weight and no date is cherry-picked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from eqresearch.config import Settings
from eqresearch.model import MODEL_FEATURES, V1_FEATURES, ExcessReturnModel, price_targets
from eqresearch.screens import SCORE_COLS

log = logging.getLogger(__name__)

STRATEGIES = ("ml_screened", "ml_only", "screens_only", "universe_ew")


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    selections: dict[pd.Timestamp, dict[str, list[str]]]
    daily: pd.DataFrame
    metrics: pd.DataFrame
    cohorts: pd.DataFrame
    ic: pd.DataFrame
    target_accuracy: pd.DataFrame
    factor_ic: pd.DataFrame
    composite_quintiles: pd.DataFrame
    ml_quintiles: pd.DataFrame
    last_date_label: pd.Timestamp


def top_n(df: pd.DataFrame, col: str, n: int, max_per_sector: int | None = None) -> list[str]:
    """Highest ``col`` first, skipping names whose sector already has ``max_per_sector`` picks."""
    ranked = df.sort_values(col, ascending=False)
    if max_per_sector is None or "sector" not in ranked:
        return ranked.index[:n].tolist()
    out, counts = [], {}
    for ticker, sector in ranked["sector"].items():
        if counts.get(sector, 0) >= max_per_sector:
            continue
        out.append(ticker)
        counts[sector] = counts.get(sector, 0) + 1
        if len(out) == n:
            break
    return out


def select(cross_section: pd.DataFrame, s: Settings) -> dict[str, list[str]]:
    cs = cross_section.droplevel("date") if isinstance(cross_section.index, pd.MultiIndex) else cross_section
    candidates = cs[cs["screens_passed"] >= s.min_screens_passed]
    cap = s.max_per_sector
    return {
        "ml_screened": top_n(candidates, "pred_excess", s.top_n, cap),
        "ml_only": top_n(cs, "pred_excess", s.top_n, cap),
        "screens_only": top_n(cs, "composite_score", s.top_n, cap),
        "universe_ew": cs.index.tolist(),
    }


def walk_forward(
    panel: pd.DataFrame, s: Settings, formation_dates: list[pd.Timestamp] | None = None, model_kwargs: dict | None = None
) -> pd.DataFrame:
    dates = sorted(panel.index.get_level_values("date").unique())
    label_date = panel.groupby(level="date")["label_date"].first()
    formation_dates = formation_dates or dates
    out = []
    for t in formation_dates:
        train_dates = [d for d in dates if pd.notna(label_date[d]) and label_date[d] <= t]
        if len(train_dates) < s.min_train_dates:
            continue
        train = panel.loc[train_dates]
        test = panel.loc[[t]]
        model = ExcessReturnModel(seed=s.seed, **(model_kwargs or {})).fit(train)
        preds = model.predict(test)
        preds = preds.join(price_targets(test["price"], preds, s.market_return))
        preds["n_train_dates"] = len(train_dates)
        preds["n_train_rows"] = len(train)
        out.append(preds)
        log.info("  %s: trained on %d dates / %d rows", t.date(), len(train_dates), len(train))
    if not out:
        raise RuntimeError("not enough history for any walk-forward fit; lower min_train_dates")
    return pd.concat(out)


def cohort_daily(prices_ff: pd.DataFrame, members: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    sub = prices_ff.loc[start:end, [m for m in members if m in prices_ff.columns]]
    sub = sub.loc[:, sub.iloc[0].notna()]
    if sub.shape[1] == 0 or len(sub) < 2:
        return pd.Series(dtype=float)
    value = (sub / sub.iloc[0]).mean(axis=1)
    return value.pct_change().iloc[1:]


def overlapping_portfolio(prices_ff, selections, strategy, horizon_days) -> pd.Series:
    last = prices_ff.index[-1]
    cohorts = {
        t: cohort_daily(prices_ff, sel[strategy], t, min(t + pd.Timedelta(days=horizon_days), last))
        for t, sel in selections.items()
    }
    frame = pd.DataFrame(cohorts)
    return frame.mean(axis=1, skipna=True).dropna()


def perf_metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty:
        return {}
    wealth = (1 + r).cumprod()
    years = len(r) / 252
    dd = wealth / wealth.cummax() - 1
    vol = r.std() * np.sqrt(252)
    return {
        "start": r.index[0].date(),
        "end": r.index[-1].date(),
        "total_return": wealth.iloc[-1] - 1,
        "cagr": wealth.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "ann_vol": vol,
        "sharpe_rf0": r.mean() * 252 / vol if vol > 0 else np.nan,
        "max_drawdown": dd.min(),
    }


def _spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < 10:
        return np.nan
    return a[ok].rank().corr(b[ok].rank())


def factor_ic(panel: pd.DataFrame, cols) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-date rank IC of each raw factor vs realised 12m excess return (no model involved)."""
    labelled = panel[panel["fwd_excess"].notna()]
    per_date = {
        d: {c: _spearman(g[c], g["fwd_excess"]) for c in cols}
        for d, g in labelled.groupby(level="date")
    }
    ics = pd.DataFrame(per_date).T
    summary = pd.DataFrame(
        {
            "mean_ic": ics.mean(),
            "ic_std": ics.std(),
            "ic_ir": ics.mean() / ics.std(),
            "pct_dates_positive": (ics > 0).where(ics.notna()).mean(),
            "n_dates": ics.count(),
        }
    ).sort_values("mean_ic", ascending=False)
    return summary, ics


def quintile_table(df: pd.DataFrame, score: str, ret: str = "fwd_return") -> pd.DataFrame:
    """Average 12m return of each score quintile minus the equal-weight universe, per date."""
    d = df[df[ret].notna() & df[score].notna()].copy()
    d["q"] = d.groupby(level="date")[score].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1)
    dates = d.index.get_level_values("date")
    d["rel"] = d[ret] - d.groupby(level="date")[ret].transform("mean")
    by_date = d.groupby([dates, "q"])["rel"].mean().unstack()
    out = pd.DataFrame({"mean_return_vs_universe": by_date.mean(), "pct_dates_beating_universe": (by_date > 0).mean()})
    out.index = [f"Q{int(i)}" for i in out.index]
    out.loc["Q5-Q1"] = [(by_date[5] - by_date[1]).mean(), ((by_date[5] - by_date[1]) > 0).mean()]
    return out


def run_backtest(panel: pd.DataFrame, prices: pd.DataFrame, s: Settings, model_kwargs: dict | None = None) -> BacktestResult:
    last_price_date = prices.index[-1]
    dates = sorted(panel.index.get_level_values("date").unique())
    # The latest close is the live scoring date, not a backtest cohort (it has no holding period).
    month_ends = [d for d in dates if d < last_price_date]

    log.info("walk-forward fits")
    preds = walk_forward(panel, s, month_ends, model_kwargs)
    joined = panel.join(preds, how="inner")

    selections = {t: select(g, s) for t, g in joined.groupby(level="date")}
    prices_ff = prices.ffill()

    daily = pd.DataFrame({k: overlapping_portfolio(prices_ff, selections, k, s.horizon_days) for k in STRATEGIES})
    first = daily.index[0]
    for b in s.benchmarks:
        if b in prices.columns:
            daily[b] = prices[b].pct_change().loc[first:]
    daily = daily.dropna(how="all")
    metrics = pd.DataFrame({k: perf_metrics(daily[k]) for k in daily.columns}).T

    # Per-cohort holding returns (to label date if matured, else to the latest close).
    rows = []
    label_date = panel.groupby(level="date")["label_date"].first()
    for t, sel in selections.items():
        end = label_date.get(t)
        matured = pd.notna(end)
        end = end if matured else last_price_date
        rel = prices_ff.loc[end] / prices.loc[t] - 1
        row = {"date": t.date(), "matured": matured, "end": end.date(), "n_universe": len(sel["universe_ew"])}
        row["n_candidates"] = int((joined.loc[t]["screens_passed"] >= s.min_screens_passed).sum())
        for k in STRATEGIES:
            row[k] = rel.reindex(sel[k]).mean()
        if "QQQ" in prices.columns:
            row["QQQ"] = rel.get("QQQ")
        rows.append(row)
    cohorts = pd.DataFrame(rows).set_index("date")

    # Information coefficient of the model, per formation date.
    ic_rows = []
    for t, g in joined.groupby(level="date"):
        matured = g["fwd_excess"].notna().any()
        if matured:
            realised = g["fwd_excess"]
        else:
            rel = prices_ff.loc[last_price_date] / prices.loc[t] - 1
            r = rel.reindex(g.index.get_level_values("ticker")).to_numpy()
            realised = pd.Series(r - np.nanmedian(r), index=g.index)
        ic_rows.append(
            {
                "date": t.date(),
                "matured": matured,
                "ml_ic": _spearman(g["pred_excess"], realised),
                "composite_ic": _spearman(g["composite_score"], realised),
            }
        )
    ic = pd.DataFrame(ic_rows).set_index("date")

    # Price-target accuracy on cohorts whose 12 months are complete.
    m = joined[joined["fwd_return"].notna()].copy()
    acc_rows = []
    if not m.empty:
        m["realised_price"] = m["price"] * (1 + m["fwd_return"])
        picks_idx = [(t, x) for t, sel in selections.items() for x in sel["ml_screened"]]
        groups = {"all eligible stocks": m, "ml_screened picks": m[m.index.isin(picks_idx)]}
        for name, g in groups.items():
            if g.empty:
                continue
            err = (g["implied_return"] - g["fwd_return"]).abs()
            naive = (s.market_return - g["fwd_return"]).abs()
            acc_rows.append(
                {
                    "group": name,
                    "n_targets": len(g),
                    "within_bear_bull_range": ((g["realised_price"] >= g["bear"]) & (g["realised_price"] <= g["bull"])).mean(),
                    "direction_hit_rate": (np.sign(g["target"] - g["price"]) == np.sign(g["fwd_return"])).mean(),
                    "realised_above_target": (g["realised_price"] >= g["target"]).mean(),
                    "median_abs_error_model": err.median(),
                    "median_abs_error_naive": naive.median(),
                    "mean_implied_return": g["implied_return"].mean(),
                    "mean_realised_return": g["fwd_return"].mean(),
                }
            )
    target_accuracy = pd.DataFrame(acc_rows).set_index("group") if acc_rows else pd.DataFrame()

    fic, _ = factor_ic(panel, [*V1_FEATURES])
    comp_q = quintile_table(panel, "composite_score")
    ml_q = quintile_table(joined, "pred_excess") if joined["fwd_excess"].notna().any() else pd.DataFrame()

    return BacktestResult(
        predictions=joined,
        selections=selections,
        daily=daily,
        metrics=metrics,
        cohorts=cohorts,
        ic=ic,
        target_accuracy=target_accuracy,
        factor_ic=fic,
        composite_quintiles=comp_q,
        ml_quintiles=ml_q,
        last_date_label=last_price_date,
    )


MODEL_VARIANTS: dict[str, dict] = {
    "v1: all features, standard trees": {"features": V1_FEATURES, "regularization": "standard"},
    "v2: fundamentals + momentum, standard trees": {"features": MODEL_FEATURES, "regularization": "standard"},
    "v3 (live): fundamentals + momentum, strong regularisation": {"features": MODEL_FEATURES, "regularization": "strong"},
}


def summarise(res: BacktestResult) -> dict:
    mat, imm = res.ic[res.ic["matured"]], res.ic[~res.ic["matured"]]
    total = (1 + res.daily.fillna(0)).prod() - 1
    return {
        "matured_ml_ic": mat["ml_ic"].mean(),
        "immature_ml_ic": imm["ml_ic"].mean(),
        "ml_screened_total": total["ml_screened"],
        "ml_only_total": total["ml_only"],
        "screens_only_total": total["screens_only"],
        "universe_ew_total": total["universe_ew"],
    }


def compare_variants(panel: pd.DataFrame, prices: pd.DataFrame, s: Settings, done: dict[str, BacktestResult] | None = None) -> pd.DataFrame:
    """Backtest every model version so the report shows all of them, not just the one kept."""
    rows = {}
    for name, kw in MODEL_VARIANTS.items():
        res = (done or {}).get(name) or run_backtest(panel, prices, s, kw)
        rows[name] = summarise(res)
    return pd.DataFrame(rows).T.rename_axis("model")


__all__ = ["run_backtest", "walk_forward", "select", "BacktestResult", "STRATEGIES", "SCORE_COLS"]
