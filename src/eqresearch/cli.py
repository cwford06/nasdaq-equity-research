"""Command line entry point: ``eqr run`` (or ``python -m eqresearch run``)."""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import replace
from pathlib import Path

from eqresearch.config import Settings


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings()
    overrides = {
        "universe": args.universe,
        "max_tickers": args.max_tickers,
        "top_n": args.top,
        "market_return": args.market_return,
        "min_mcap": args.min_mcap * 1e9 if args.min_mcap is not None else None,
        "min_screens_passed": args.min_screens,
        "workers": args.workers,
        "reports_dir": Path(args.out) if args.out else None,
        "max_per_sector": args.max_per_sector,
    }
    s = replace(s, **{k: v for k, v in overrides.items() if v is not None})
    if args.max_per_sector == 0:
        s = replace(s, max_per_sector=None)
    if args.include_financials:
        s = replace(s, exclude_sectors=())
    return s


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eqr", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_ in (
        ("run", "download data, backtest, and write picks + reports"),
        ("fetch", "download / refresh the Yahoo Finance cache only"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--universe", choices=("nasdaq", "nasdaq100"))
        p.add_argument("--max-tickers", type=int)
        p.add_argument("--top", type=int, help="number of picks (default 20)")
        p.add_argument("--market-return", type=float, help="assumed 12m median-stock return (default 0.08)")
        p.add_argument("--min-mcap", type=float, help="minimum market cap in $bn (default 1)")
        p.add_argument("--min-screens", type=int, choices=(0, 1, 2, 3), help="pillar screens a pick must pass (default 2)")
        p.add_argument("--max-per-sector", type=int, help="cap on picks per sector (default 5, 0 = no cap)")
        p.add_argument("--include-financials", action="store_true", help="keep Finance and Real Estate stocks")
        p.add_argument("--workers", type=int)
        p.add_argument("--out", help="report folder (default reports/)")
        p.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    s = _settings(args)

    from eqresearch import pipeline

    if args.cmd == "fetch":
        pipeline.load_dataset(s)
        return

    result = pipeline.run_all(s)
    res = result["backtest"]
    print("\nBacktest (walk-forward, overlapping 12m cohorts):")
    print(res.metrics[["start", "end", "total_return", "cagr", "sharpe_rf0", "max_drawdown"]].to_string(
        formatters={c: "{:.1%}".format for c in ("total_return", "cagr", "max_drawdown")} | {"sharpe_rf0": "{:.2f}".format}))
    picks = result["picks"]
    print(f"\nTop {len(picks)} picks:")
    print(picks[["price", "target", "bear", "bull", "implied_return"]].to_string(
        formatters={"price": "${:,.2f}".format, "target": "${:,.2f}".format, "bear": "${:,.0f}".format,
                    "bull": "${:,.0f}".format, "implied_return": "{:+.0%}".format}))
    for p in result["paths"]:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
