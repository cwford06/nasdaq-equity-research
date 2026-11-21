"""End-to-end orchestration: data -> features -> backtest -> live picks -> reports."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from eqresearch.backtest import MODEL_VARIANTS, BacktestResult, compare_variants, run_backtest, select
from eqresearch.config import Settings
from eqresearch.data import fetch_fundamentals, fetch_prices, fetch_profiles
from eqresearch.features import build_panel, fiscal_features
from eqresearch.fundamentals import annual_panel, duplicate_share_classes, quarterly_trends
from eqresearch.model import ExcessReturnModel, price_targets
from eqresearch.report import write_backtest_report, write_picks
from eqresearch.screens import add_pillar_scores
from eqresearch.universe import load_universe, sectors

log = logging.getLogger(__name__)


class DataQualityError(RuntimeError):
    """Raised in strict mode when the downloaded data is too incomplete to publish picks from."""


@dataclass
class Dataset:
    tickers: list[str]
    raw: dict
    prices: pd.DataFrame
    panel: pd.DataFrame


def load_dataset(s: Settings) -> Dataset:
    tickers = load_universe(s.universe, s.max_tickers)
    sector = sectors(s.cache_dir)
    if s.strict and s.exclude_sectors and sector.empty:
        raise DataQualityError("sector lookup failed, so Finance/Real Estate cannot be excluded")
    if s.exclude_sectors and not sector.empty:
        before = len(tickers)
        tickers = [t for t in tickers if sector.get(t, "") not in s.exclude_sectors]
        log.info("excluded %d tickers in sectors %s", before - len(tickers), ", ".join(s.exclude_sectors))
    log.info("universe %s: %d tickers", s.universe, len(tickers))
    raw = fetch_fundamentals(tickers, s.cache_dir, s.fundamentals_max_age_days, s.workers)
    coverage = len(raw) / max(len(tickers), 1)
    log.info("fundamentals available for %d tickers (%.0f%% of universe)", len(raw), coverage * 100)
    if s.strict and coverage < s.min_fundamentals_coverage:
        raise DataQualityError(
            f"fundamentals for only {coverage:.0%} of tickers (need {s.min_fundamentals_coverage:.0%}); "
            "Yahoo is probably rate limiting"
        )
    prices = fetch_prices(sorted(set(tickers) | set(s.benchmarks)), s.price_start, s.cache_dir, s.prices_max_age_hours)
    annual = annual_panel(raw)
    dupes = duplicate_share_classes(annual)
    if dupes:
        log.info("dropping %d duplicate share classes: %s", len(dupes), ", ".join(dupes))
        annual = annual[~annual["ticker"].isin(dupes)]
        raw = {t: v for t, v in raw.items() if t not in dupes}
    fiscal = fiscal_features(annual, s.report_lag_days)
    panel = add_pillar_scores(build_panel(prices, fiscal, s, tickers=[t for t in tickers if t in raw]))
    panel["sector"] = panel.index.get_level_values("ticker").map(lambda t: sector.get(t, "") or "Unknown")
    dates = panel.index.get_level_values("date")
    log.info("feature panel: %d rows, %d dates (%s to %s)", len(panel), dates.nunique(), dates.min().date(), dates.max().date())
    return Dataset(tickers, raw, prices, panel)


def live_picks(ds: Dataset, s: Settings) -> tuple[pd.DataFrame, pd.Series, pd.Timestamp, pd.DataFrame]:
    """Fit on every labelled row, then score and pick from the latest cross-section."""
    panel = ds.panel
    as_of = panel.index.get_level_values("date").max()
    model = ExcessReturnModel(seed=s.seed).fit(panel)
    cs = panel.loc[[as_of]]
    preds = model.predict(cs)
    cs = cs.join(preds).join(price_targets(cs["price"], preds, s.market_return)).droplevel("date")
    chosen = select(cs, s)["ml_screened"]
    picks = cs.loc[chosen].copy()

    picks = picks.join(quarterly_trends({t: ds.raw[t] for t in chosen if t in ds.raw}))
    profiles = fetch_profiles(chosen, s.workers)
    picks["name"] = profiles["longName"].fillna(profiles["shortName"])
    picks["sector"] = picks["sector"].where(picks["sector"] != "Unknown", profiles["sector"])
    picks["analyst_target"] = pd.to_numeric(profiles["targetMeanPrice"], errors="coerce")
    importance = model.importance()
    return picks, importance, as_of, cs


def run_all(s: Settings) -> dict:
    ds = load_dataset(s)
    log.info("backtesting")
    res: BacktestResult = run_backtest(ds.panel, ds.prices, s)
    log.info("backtesting earlier model versions for comparison")
    live_name = next(k for k in MODEL_VARIANTS if "(live)" in k)
    variants = compare_variants(ds.panel, ds.prices, s, done={live_name: res})
    log.info("scoring live cross-section")
    picks, importance, as_of, scored = live_picks(ds, s)
    out = s.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    keep = ["price", "target", "bear", "bull", "implied_return", "pred_excess", "growth_score", "margin_score",
            "value_score", "composite_score", "screens_passed", "rev_growth", "d_op_margin", "ev_sales", "fcf_yield", "mcap"]
    scored[keep].sort_values("pred_excess", ascending=False).to_csv(out / "universe_scores_latest.csv")
    v1 = MODEL_VARIANTS[next(k for k in MODEL_VARIANTS if k.startswith("v1"))]
    v1_importance = ExcessReturnModel(seed=s.seed, **v1).fit(ds.panel).importance(n_repeats=3)
    bt_path = write_backtest_report(res, s, len(ds.tickers), importance, out, variants, v1_importance)
    picks_path = write_picks(picks, s, as_of, list(importance.head(5).index), out)
    return {"backtest": res, "variants": variants, "picks": picks, "importance": importance, "paths": (bt_path, picks_path)}
