"""Run settings shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # Universe
    universe: str = "nasdaq"  # "nasdaq" (Global Select + Global Market common stock) or "nasdaq100"
    max_tickers: int | None = None  # cap the universe (useful for quick runs)
    # Revenue growth, gross margin, EV/Sales and FCF do not describe banks, insurers,
    # BDCs or REITs, so (as in most fundamental factor research) they are excluded.
    exclude_sectors: tuple[str, ...] = ("Finance", "Real Estate")
    max_per_sector: int | None = 5  # cap on picks from one sector (None = no cap)

    # Data
    price_start: str = "2022-01-01"  # needs >= 1y of history before the first as-of date
    scan_from: str = "2023-01-31"  # earliest as-of date considered; earlier dates lack 2 statements
    report_lag_days: int = 90  # a fiscal year is only "known" this many days after it ends
    max_staleness_days: int = 550  # ignore fundamentals older than this at the as-of date
    cache_dir: Path = Path("data/cache")
    fundamentals_max_age_days: int = 7
    prices_max_age_hours: float = 12.0
    workers: int = 4
    benchmarks: tuple[str, ...] = ("QQQ",)

    # Eligibility filters, applied point-in-time
    min_mcap: float = 1e9
    min_price: float = 5.0
    min_history_days: int = 252
    min_cross_section: int = 25  # skip as-of dates with fewer eligible stocks than this

    # Model / targets
    horizon_days: int = 365
    market_return: float = 0.08  # assumed 12m return of the median stock, added to predicted excess
    min_train_dates: int = 6  # monthly as-of dates with fully realised labels before the first fit
    top_n: int = 20
    min_screens_passed: int = 2  # of the 3 pillar screens (growth, margin, value)
    seed: int = 0

    reports_dir: Path = field(default=Path("reports"))
