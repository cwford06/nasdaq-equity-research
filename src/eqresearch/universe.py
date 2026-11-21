"""Which tickers to research.

Yahoo Finance does not publish index membership, so the ticker list comes from
Nasdaq itself: the Nasdaq-100 constituent API, or the Nasdaq Trader symbol
directory for all Nasdaq listings. All prices and fundamentals come from Yahoo.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (equity-research script)"}
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
}
_FALLBACK = Path(__file__).with_name("nasdaq100_fallback.txt")
_EXCLUDE_WORDS = ("warrant", "right", " unit", "units", "preferred", "notes due", "debenture", "%")


def to_yahoo(symbol: str) -> str:
    return symbol.strip().upper().replace(".", "-")


def nasdaq100() -> list[str]:
    try:
        resp = requests.get(
            "https://api.nasdaq.com/api/quote/list-type/nasdaq100",
            headers={**_BROWSER_HEADERS, "Accept": "application/json"},
            timeout=20,
        )
        resp.raise_for_status()
        rows = resp.json()["data"]["data"]["rows"]
        tickers = sorted({to_yahoo(r["symbol"]) for r in rows})
        if not 90 <= len(tickers) <= 110:
            raise ValueError(f"unexpected constituent count {len(tickers)}")
        return tickers
    except Exception as exc:  # network or API change
        log.warning("Falling back to bundled Nasdaq-100 list (%s)", exc)
        return sorted(line.strip() for line in _FALLBACK.read_text().splitlines() if line.strip())


def nasdaq_listed(categories: tuple[str, ...] = ("Q", "G")) -> list[str]:
    """Common stocks on the Nasdaq Global Select (Q) and Global Market (G) tiers."""
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    text = requests.get(url, headers=_HEADERS, timeout=30).text
    df = pd.read_csv(io.StringIO(text), sep="|")
    df = df[~df["Symbol"].astype(str).str.startswith("File Creation")]
    name = df["Security Name"].astype(str).str.lower()
    keep = (
        (df["Test Issue"] == "N")
        & (df["ETF"] == "N")
        & (df["Financial Status"].fillna("N") == "N")
        & df["Market Category"].isin(categories)
        & ~name.apply(lambda s: any(w in s for w in _EXCLUDE_WORDS))
        & df["Symbol"].astype(str).str.fullmatch(r"[A-Z]{1,5}")
    )
    return sorted({to_yahoo(s) for s in df.loc[keep, "Symbol"]})


def sectors(cache_dir: Path | None = None, max_age_days: float = 7) -> pd.Series:
    """Nasdaq's sector label for every Nasdaq listing (one request), cached for a week."""
    path = Path(cache_dir) / "sectors.csv" if cache_dir else None
    if path and path.exists() and (pd.Timestamp.now().timestamp() - path.stat().st_mtime) < max_age_days * 86400:
        return pd.read_csv(path, index_col=0, keep_default_na=False)["sector"]
    try:
        resp = requests.get(
            "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange=nasdaq&download=true",
            headers={**_BROWSER_HEADERS, "Accept": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        rows = resp.json()["data"]["rows"]
    except Exception as exc:
        log.warning("sector lookup failed (%s); no sector exclusions or caps will apply", exc)
        return pd.Series(dtype=str, name="sector")
    out = pd.Series({to_yahoo(r["symbol"]): (r.get("sector") or "").strip() for r in rows}, name="sector")
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.rename_axis("ticker").to_csv(path)
    return out


def load_universe(kind: str, max_tickers: int | None = None) -> list[str]:
    if kind == "nasdaq100":
        tickers = nasdaq100()
    elif kind == "nasdaq":
        tickers = nasdaq_listed()
    else:
        raise ValueError(f"unknown universe {kind!r}; use 'nasdaq100' or 'nasdaq'")
    return tickers[:max_tickers] if max_tickers else tickers
