"""Markdown and chart output."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eqresearch.backtest import BacktestResult  # noqa: E402
from eqresearch.config import Settings  # noqa: E402

STRATEGY_LABELS = {
    "ml_screened": "ML + screens (headline)",
    "ml_only": "ML only",
    "screens_only": "Screens only (no ML)",
    "universe_ew": "Equal-weight universe",
    "QQQ": "QQQ",
}
COLORS = {
    "ml_screened": "#1f5fbf", "ml_only": "#7a9fd9", "screens_only": "#d98a1f",
    "universe_ew": "#7f7f7f", "QQQ": "#222222",
}


def _pct(x, digits=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{digits}f}%"


def _md_table(df: pd.DataFrame, pct_cols=(), num_cols=(), digits=2) -> str:
    d = df.copy()
    for c in d.columns:
        if c in pct_cols:
            d[c] = d[c].map(lambda v: _pct(v))
        elif c in num_cols:
            d[c] = d[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.{digits}f}")
    d = d.reset_index()
    if d.columns[0] == "index":
        d = d.rename(columns={"index": ""})
    header = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


_ACCURACY_LABELS = {
    "n_targets": "Targets evaluated",
    "within_bear_bull_range": "Realised price inside bear-bull range",
    "direction_hit_rate": "Direction correct (up vs down)",
    "realised_above_target": "Realised price at or above target",
    "median_abs_error_model": "Median abs. error, model (return pts)",
    "median_abs_error_naive": "Median abs. error, naive flat target",
    "mean_implied_return": "Mean implied return",
    "mean_realised_return": "Mean realised return",
}


def _format_accuracy(ta: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for col, label in _ACCURACY_LABELS.items():
        rows[label] = ta[col].map(lambda v: f"{int(v)}" if col == "n_targets" else _pct(v))
    return pd.DataFrame(rows).T.rename_axis("metric")


def equity_chart(daily: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col in daily.columns:
        wealth = (1 + daily[col].fillna(0)).cumprod()
        ax.plot(wealth.index, wealth, label=STRATEGY_LABELS.get(col, col), color=COLORS.get(col),
                lw=2.2 if col == "ml_screened" else 1.3)
    ax.axhline(1, color="#bbbbbb", lw=0.8)
    ax.set_ylabel("Growth of $1")
    ax.set_title("Walk-forward backtest: overlapping 12-month cohorts, formed monthly")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def factor_ic_chart(fic: pd.DataFrame, path: Path) -> None:
    d = fic["mean_ic"].sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(d) + 1))
    ax.barh(d.index, d.values, color=np.where(d.values >= 0, "#1f5fbf", "#c0504d"))
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_xlabel("Mean rank IC vs realised 12m excess return")
    ax.set_title("Which signals predicted 12-month returns (no model)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def key_findings(res: BacktestResult, s: Settings) -> list[str]:
    """Plain-language summary whose wording follows the numbers, so it stays true on re-runs."""
    m = res.metrics
    head, univ = m.loc["ml_screened", "total_return"], m.loc["universe_ew", "total_return"]
    qqq = m.loc["QQQ", "total_return"] if "QQQ" in m.index else np.nan
    n_mat = int(res.cohorts["matured"].sum())
    ic = res.ic["ml_ic"]
    out = [
        f"- **Headline strategy:** {_pct(head)} vs {_pct(univ)} for the equal-weight universe and {_pct(qqq)} for "
        f"QQQ, from {m.loc['ml_screened', 'start']} to {m.loc['ml_screened', 'end']} ({len(res.cohorts)} monthly "
        f"cohorts, {n_mat} with a full 12-month outcome)."
    ]
    rank_txt = (
        f"- **Ranking skill:** the ML forecast's rank IC across the whole eligible universe averaged {ic.mean():+.3f} "
        f"and was positive on {(ic > 0).mean():.0%} of formation dates."
    )
    if head > univ and ic.mean() <= 0:
        rank_txt += (
            " So the top-20 outperformance did **not** come from the model ranking the universe well. It rests on "
            "a handful of names that passed the screens, and at this sample size it cannot be told apart from luck."
        )
    elif head > univ:
        rank_txt += " The ranking skill and the portfolio result point the same way, but the sample is short."
    else:
        rank_txt += " The model did not add value over the equal-weight universe in this sample."
    out.append(rank_txt)
    out.append(f"- **Rule-based screens alone:** mean rank IC {res.ic['composite_ic'].mean():+.3f}.")
    if not res.target_accuracy.empty and "all eligible stocks" in res.target_accuracy.index:
        ta = res.target_accuracy.loc["all eligible stocks"]
        better = ta["median_abs_error_model"] < ta["median_abs_error_naive"]
        out.append(
            f"- **Price targets:** median absolute error {_pct(ta['median_abs_error_model'])} vs "
            f"{_pct(ta['median_abs_error_naive'])} for a flat +{s.market_return:.0%} target, so the model's targets were "
            f"{'more' if better else 'no more'} accurate than the naive baseline. "
            f"{_pct(ta['within_bear_bull_range'], 0)} of realised prices landed inside the bear-bull range "
            "(nominally 80%)."
        )
    return out


def write_backtest_report(
    res: BacktestResult, s: Settings, universe_size: int, importance: pd.Series, out: Path,
    variants: pd.DataFrame | None = None, v1_importance: pd.Series | None = None,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    equity_chart(res.daily, out / "equity_curve.png")
    factor_ic_chart(res.factor_ic, out / "factor_ic.png")
    res.cohorts.to_csv(out / "backtest_cohorts.csv")
    res.daily.to_csv(out / "backtest_daily_returns.csv")
    res.factor_ic.to_csv(out / "factor_ic.csv")
    cols = ["price", "pred_excess", "target", "bear", "bull", "implied_return", "fwd_return",
            "growth_score", "margin_score", "value_score", "composite_score", "screens_passed"]
    res.predictions[cols].to_csv(out / "backtest_predictions.csv")

    m = res.metrics.rename(index=STRATEGY_LABELS)
    matured = res.ic[res.ic["matured"]]
    immature = res.ic[~res.ic["matured"]]
    n_form = len(res.cohorts)
    first, last = res.cohorts.index[0], res.cohorts.index[-1]

    lines = [
        "# Backtest report",
        "",
        f"Generated from Yahoo Finance data through **{res.last_date_label.date()}**. "
        f"Universe: `{s.universe}` ({universe_size} tickers requested; {int(res.cohorts['n_universe'].median())} "
        f"eligible in a typical month after the ${s.min_mcap / 1e9:.0f}B market-cap, ${s.min_price:.0f} price "
        "and data-availability filters). Finance and Real Estate stocks are excluded because revenue, "
        "margin and EV-based metrics do not describe lenders or REITs.",
        "",
        "## Key findings",
        "",
        *key_findings(res, s),
        "",
        "## Setup",
        "",
        f"- **Formation dates:** {n_form} month-ends from {first} to {last}. On each one the model is refit "
        f"using only samples whose 12-month outcome was fully known by that date "
        f"(minimum {s.min_train_dates} monthly cross-sections).",
        f"- **Portfolios:** top {s.top_n} names, equal weight, held 12 months. Monthly cohorts overlap, and the "
        "strategy return is the average of all live cohorts.",
        f"- **Headline strategy (ML + screens):** stocks passing at least {s.min_screens_passed} of the 3 pillar "
        "screens (revenue growth, margin expansion, valuation), ranked by the ML forecast.",
        f"- **Price target:** price x (1 + {s.market_return:.0%} assumed median-stock return) x exp(predicted log "
        "excess return). Bear and bull cases come from 10th and 90th percentile models.",
        "- **Costs:** none are modelled. A 12-month holding period keeps turnover low, but subtract roughly "
        "0.1-0.3% a year for large caps.",
        "",
        "## Strategy performance",
        "",
        "![equity curve](equity_curve.png)",
        "",
        _md_table(m.rename_axis("strategy"), pct_cols=("total_return", "cagr", "ann_vol", "max_drawdown"), num_cols=("sharpe_rf0",)),
        "",
        f"Cohorts formed after {res.cohorts[res.cohorts['matured']].index.max() if res.cohorts['matured'].any() else 'n/a'} "
        "have not finished their 12 months yet. Their returns run to the latest close, so the last part of the "
        "curve uses partial holding periods.",
        "",
        "### Per-cohort returns (formation date to 12 months later, or to date)",
        "",
        _md_table(res.cohorts.rename(columns=STRATEGY_LABELS), pct_cols=[STRATEGY_LABELS[k] for k in STRATEGY_LABELS]),
        "",
        "## Does the model rank stocks correctly?",
        "",
        "Rank information coefficient (IC) is the Spearman correlation between the forecast and the realised "
        "return across stocks on a formation date. +1 is a perfect ranking and 0 is no skill. Monthly "
        "cross-sectional ICs of 0.03-0.08 are considered useful in practice. For calibration: on pure-noise "
        "synthetic data with the same structure, this backtest produces mean ICs between about -0.05 and +0.10, "
        "so values in that range cannot be told apart from luck.",
        "",
        f"- Matured cohorts (full 12 months, n={len(matured)}): mean ML IC **{matured['ml_ic'].mean():.3f}**, "
        f"mean screens-only IC {matured['composite_ic'].mean():.3f}" if len(matured) else "- No matured cohorts yet.",
        f"- Immature cohorts (returns to date, n={len(immature)}): mean ML IC {immature['ml_ic'].mean():.3f}, "
        f"mean screens-only IC {immature['composite_ic'].mean():.3f}" if len(immature) else "",
        "",
        _md_table(res.ic, num_cols=("ml_ic", "composite_ic"), digits=3),
        "",
    ]
    if not res.ml_quintiles.empty:
        lines += [
            "ML forecast quintiles on matured cohorts (Q5 = highest forecast), as average 12m return minus the "
            "equal-weight universe:",
            "",
            _md_table(res.ml_quintiles.rename_axis("quintile"), pct_cols=("mean_return_vs_universe", "pct_dates_beating_universe")),
            "",
        ]
    if not res.target_accuracy.empty:
        ta = res.target_accuracy
        lines += [
            "## Price-target accuracy (matured cohorts)",
            "",
            "Compares each 12-month target with the price actually reached. The naive baseline gives every "
            f"stock the same +{s.market_return:.0%} target.",
            "",
            _md_table(_format_accuracy(ta)),
            "",
        ]
    if variants is not None and not variants.empty:
        lines += [
            "## Model versions tried",
            "",
            "Every model version is backtested on every run, so the version kept for live picks can be compared "
            "with the ones dropped. v1 also used size, volatility and short-term price features. During development, "
            "on a partial download (about 1,160 tickers, before rate-limited requests were retried), v1 leaned "
            "mainly on size and volatility and picked a cluster of small, volatile semiconductor stocks"
            + (f". On this run's data its most important features are {', '.join(v1_importance.head(5).index)}"
               if v1_importance is not None else "")
            + ". v2 and v3 drop those features so the model stays about company fundamentals (revenue growth, "
            "margins, valuation) plus 12-1 month momentum, and v3 adds stronger regularisation because the "
            "training sample is small. These changes were made *after* seeing v1's backtest, so v3's numbers "
            "carry some selection bias. The fair reading is the whole table, not just the best row.",
            "",
            _md_table(
                variants,
                num_cols=("matured_ml_ic", "immature_ml_ic"),
                pct_cols=("ml_screened_total", "ml_only_total", "screens_only_total", "universe_ew_total"),
                digits=3,
            ),
            "",
        ]
    lines += [
        "## Which research signals worked? (whole labelled sample, no model)",
        "",
        "Mean monthly rank IC of each raw signal against realised 12-month excess return, across every "
        f"as-of date with a complete 12-month outcome ({int(res.factor_ic['n_dates'].max())} dates). "
        "Overlapping 12-month windows mean neighbouring months are not independent, so treat this as "
        "descriptive.",
        "",
        "![factor IC](factor_ic.png)",
        "",
        _md_table(res.factor_ic.rename_axis("signal"), num_cols=("mean_ic", "ic_std", "ic_ir"), pct_cols=("pct_dates_positive",), digits=3),
        "",
        "Composite (screens-only) score quintiles over the same sample:",
        "",
        _md_table(res.composite_quintiles.rename_axis("quintile"), pct_cols=("mean_return_vs_universe", "pct_dates_beating_universe")),
        "",
        "## What the final model leans on",
        "",
        "Permutation importance (drop in in-sample R^2 when a feature is shuffled) for the model used for the "
        "live picks:",
        "",
        _md_table(importance.head(12).rename("importance").rename_axis("feature").to_frame(), num_cols=("importance",), digits=4),
        "",
        "## Caveats",
        "",
        "- **Short history.** Yahoo serves only the last four fiscal years of statements, so the point-in-time "
        "backtest starts in 2024 and only a few cohorts have a full 12-month outcome. Read the results as a "
        "sanity check, not proof of an edge.",
        "- **Survivorship bias.** The universe is today's Nasdaq listings, so companies that delisted or went "
        "bankrupt since 2024 are missing. This flatters every strategy, including the equal-weight benchmark.",
        "- **Restated fundamentals.** Yahoo shows the latest restated figures, not what was first reported. "
        f"The {s.report_lag_days}-day reporting lag guards against look-ahead on timing, but not on revisions.",
        "- **One market regime.** 2024-2026 was dominated by a narrow group of AI and semiconductor winners. "
        "Signals that worked or failed here may behave differently in another regime.",
        "",
    ]
    path = out / "backtest.md"
    path.write_text("\n".join(lines))
    return path


def _note(r: pd.Series) -> str:
    parts = []
    g = r.get("rev_growth")
    if pd.notna(g):
        verb = f"grew {_pct(g, 0)}" if g >= 0 else f"fell {_pct(-g, 0)}"
        trend = ""
        if pd.notna(r.get("rev_growth_prev")):
            better = g > r["rev_growth_prev"]
            word = ("accelerating" if better else "slowing") if g >= 0 else ("improving" if better else "worsening")
            trend = f" ({word} from {_pct(r['rev_growth_prev'], 0)} the year before)"
        parts.append(f"Revenue {verb} in the last fiscal year{trend}.")
    if pd.notna(r.get("q_rev_growth_yoy")):
        parts.append(f"Latest quarter ({r['latest_quarter']}): revenue {_pct(r['q_rev_growth_yoy'], 0)} YoY.")
    d = r.get("d_op_margin")
    if pd.notna(d):
        verb = "expanded" if d >= 0 else "contracted"
        size = "more than 100pp" if abs(d) >= 0.999 else f"{abs(d) * 100:.1f}pp"  # feature is clipped at +-100pp
        parts.append(f"Operating margin {verb} {size} to {_pct(r['op_margin'], 1)}.")
    val = []
    if pd.notna(r.get("ev_sales")):
        val.append(f"EV/Sales {r['ev_sales']:.1f}x")
    if pd.notna(r.get("pe")) and r["pe"] > 0:
        val.append(f"P/E {r['pe']:.0f}x")
    if pd.notna(r.get("fcf_yield")):
        val.append(f"FCF yield {_pct(r['fcf_yield'], 1)}")
    if val:
        parts.append("Valuation: " + ", ".join(val) + ".")
    return " ".join(parts)


def write_picks(picks: pd.DataFrame, s: Settings, as_of: pd.Timestamp, importance_top: list[str], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    stamp = as_of.date().isoformat()
    csv_cols = [
        "name", "sector", "price", "target", "bear", "bull", "implied_return", "analyst_target",
        "growth_score", "margin_score", "value_score", "composite_score", "screens_passed",
        "rev_growth", "rev_accel", "q_rev_growth_yoy", "op_margin", "d_op_margin", "d_gross_margin",
        "ev_sales", "ev_sales_to_growth", "pe", "fcf_yield", "mcap",
    ]
    picks[[c for c in csv_cols if c in picks.columns]].to_csv(out / f"picks_{stamp}.csv")
    picks[[c for c in csv_cols if c in picks.columns]].to_csv(out / "picks_latest.csv")

    table = pd.DataFrame(
        {
            "Name": picks["name"].fillna("").str.slice(0, 34),
            "Price": picks["price"].map(lambda v: f"${v:,.2f}"),
            "12m target": picks["target"].map(lambda v: f"${v:,.2f}"),
            "Bear / Bull": [f"${b:,.0f} / ${u:,.0f}" for b, u in zip(picks["bear"], picks["bull"])],
            "Upside": picks["implied_return"].map(lambda v: _pct(v, 0)),
            "Analyst mean": picks["analyst_target"].map(lambda v: "n/a" if pd.isna(v) else f"${v:,.0f}"),
            "Growth / Margin / Value": [
                f"{g:.2f} / {m:.2f} / {v:.2f}" for g, m, v in zip(picks["growth_score"], picks["margin_score"], picks["value_score"])
            ],
        },
        index=picks.index,
    )
    lines = [
        f"# Nasdaq picks as of {stamp}",
        "",
        f"The top {len(picks)} stocks that pass at least {s.min_screens_passed} of the 3 research screens "
        "(revenue growth, margin expansion, valuation), ranked by the ML model's 12-month excess-return forecast.",
        "",
        f"Targets assume the median stock returns {s.market_return:.0%} over 12 months. Bear and bull cases are the "
        "model's 10th and 90th percentile outcomes. Pillar scores are percentile ranks within the eligible "
        "universe (1.00 = best). The analyst mean is Yahoo's consensus target, shown for comparison only and "
        "not used by the model.",
        "",
        f"Signals the model leans on most: {', '.join(importance_top)}.",
        "",
        _md_table(table),
        "",
        "## Research notes",
        "",
    ]
    for t, r in picks.iterrows():
        lines += [
            f"### {t}: {r.get('name') or ''}",
            f"*{r.get('sector') or 'n/a'}* · target **${r['target']:,.2f}** ({_pct(r['implied_return'], 0)}), "
            f"range ${r['bear']:,.0f} to ${r['bull']:,.0f}",
            "",
            _note(r),
            "",
        ]
    lines += [
        "---",
        "Generated by a quantitative model for research purposes. This is not investment advice. See "
        "[backtest.md](backtest.md) for how the model has performed and its limitations.",
    ]
    path = out / "picks_latest.md"
    path.write_text("\n".join(lines))
    (out / f"picks_{stamp}.md").write_text("\n".join(lines))
    return path
