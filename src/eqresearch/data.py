"""Yahoo Finance downloads with an on-disk cache and rate-limit backoff."""

from __future__ import annotations

import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

_STATEMENTS = {
    "income": lambda t: t.get_income_stmt(freq="yearly"),
    "balance": lambda t: t.get_balance_sheet(freq="yearly"),
    "cashflow": lambda t: t.get_cash_flow(freq="yearly"),
    "income_q": lambda t: t.get_income_stmt(freq="quarterly"),
}


def _retry(fn, attempts: int = 4, base_wait: float = 2.0):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            wait = base_wait * 2**attempt
            log.debug("retrying in %.0fs after %s", wait, exc)
            time.sleep(wait)


def _is_fresh(path: Path, max_age_seconds: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_seconds


EMPTY = "EMPTY_CONFIRMED"  # cache marker: Yahoo has no statements for this ticker (SPACs, shells)
_CANARY = "MSFT"


def _fetch_one(ticker: str) -> dict[str, pd.DataFrame] | None:
    tk = yf.Ticker(ticker)
    income = _retry(lambda: _STATEMENTS["income"](tk))
    if income is None or income.empty:
        return None
    out = {"income": income}
    out |= {name: _retry(lambda f=f: f(tk)) for name, f in _STATEMENTS.items() if name != "income"}
    return out


def _yahoo_responsive() -> bool:
    """yfinance returns empty tables (not errors) when rate limited, so probe a known-good ticker."""
    try:
        return not yf.Ticker(_CANARY).get_income_stmt(freq="yearly").empty
    except Exception:
        return False


def _wait_for_yahoo(max_wait: float = 900) -> bool:
    waited, pause = 0.0, 30.0
    while not _yahoo_responsive():
        if waited >= max_wait:
            return False
        log.info("Yahoo is rate limiting; waiting %.0fs", pause)
        time.sleep(pause)
        waited += pause
        pause = min(pause * 2, 240)
    return True


def _download(tickers: list[str], workers: int) -> tuple[dict, list[str], list[str]]:
    got, empty, failed = {}, [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                data = fut.result()
            except Exception as exc:
                log.debug("fundamentals failed for %s: %s", t, exc)
                failed.append(t)
                continue
            if data is None:
                empty.append(t)
            else:
                got[t] = data
            if i % 200 == 0:
                log.info("  %d/%d", i, len(tickers))
    return got, empty, failed


def fetch_fundamentals(
    tickers: list[str], cache_dir: Path, max_age_days: float = 7, workers: int = 4, max_rounds: int = 4
) -> dict[str, dict[str, pd.DataFrame]]:
    """Annual income/balance/cash-flow statements plus quarterly income, per ticker.

    Empty answers are only trusted (and cached) when the canary ticker loads fine
    right after; otherwise they are retried in later, slower rounds.
    """
    folder = Path(cache_dir) / "fundamentals"
    folder.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, pd.DataFrame]] = {}
    todo = []
    for t in tickers:
        path = folder / f"{t}.pkl"
        cached = None
        if _is_fresh(path, max_age_days * 86400):
            with path.open("rb") as fh:
                cached = pickle.load(fh)
        if isinstance(cached, dict):
            result[t] = cached
        elif cached != EMPTY:
            todo.append(t)

    if todo:
        log.info("downloading fundamentals for %d tickers (%d cached)", len(todo), len(result))
    for rnd in range(max_rounds):
        if not todo:
            break
        if rnd and not _wait_for_yahoo():
            break
        got, empty, failed = _download(todo, workers if rnd == 0 else max(1, workers // 2))
        for t, data in got.items():
            with (folder / f"{t}.pkl").open("wb") as fh:
                pickle.dump(data, fh)
        result |= got
        if empty and _yahoo_responsive():
            # Yahoo answered the canary, so these really have no statements... unless the
            # limit kicked in mid-batch. Re-check large batches once more before trusting them.
            if rnd > 0 or len(empty) < 50:
                for t in empty:
                    with (folder / f"{t}.pkl").open("wb") as fh:
                        pickle.dump(EMPTY, fh)
                empty = []
        todo = empty + failed
        if todo:
            log.info("round %d: %d downloaded, %d to retry", rnd + 1, len(got), len(todo))
    if todo:
        log.warning("fundamentals unavailable for %d tickers after retries: %s", len(todo), ", ".join(todo[:20]))
    return result


def fetch_prices(
    tickers: list[str], start: str, cache_dir: Path, max_age_hours: float = 12, chunk: int = 200
) -> pd.DataFrame:
    """Split- and dividend-adjusted daily closes, one column per ticker."""
    path = Path(cache_dir) / "prices.parquet"
    if _is_fresh(path, max_age_hours * 3600):
        cached = pd.read_parquet(path)
        if set(tickers) <= set(cached.columns) and str(cached.index[0].date()) <= start:
            return cached[sorted(tickers)]

    frames = []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        raw = _retry(
            lambda b=batch: yf.download(
                b, start=start, auto_adjust=True, progress=False, threads=True, group_by="column"
            )
        )
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(batch[0])
        frames.append(close)
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()].dropna(axis=1, how="all").sort_index()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).astype("datetime64[ns]")
    prices.columns = [str(c) for c in prices.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    return prices


def fetch_profiles(tickers: list[str], workers: int = 4) -> pd.DataFrame:
    """Current name, sector and consensus analyst target, for the research notes only."""
    keys = ("longName", "shortName", "sector", "industry", "targetMeanPrice", "numberOfAnalystOpinions", "recommendationKey")
    _wait_for_yahoo(max_wait=600)

    def one(t: str) -> dict:
        try:
            info = _retry(lambda: yf.Ticker(t).info, attempts=3)
        except Exception:
            info = {}
        return {"ticker": t, **{k: info.get(k) for k in keys}}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, tickers))
    return pd.DataFrame(rows).set_index("ticker")
