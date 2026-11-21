import pandas as pd
import pytest

from eqresearch import pipeline
from eqresearch.config import Settings


@pytest.fixture
def offline(monkeypatch):
    tickers = [f"T{i}" for i in range(10)]
    monkeypatch.setattr(pipeline, "load_universe", lambda *a, **k: list(tickers))
    monkeypatch.setattr(pipeline, "sectors", lambda *a, **k: pd.Series({t: "Technology" for t in tickers}))
    monkeypatch.setattr(pipeline, "fetch_fundamentals", lambda tk, *a, **k: {t: {} for t in tk})
    return monkeypatch


def test_strict_fails_when_sector_lookup_fails(offline):
    offline.setattr(pipeline, "sectors", lambda *a, **k: pd.Series(dtype=str))
    with pytest.raises(pipeline.DataQualityError, match="sector"):
        pipeline.load_dataset(Settings(strict=True))


def test_strict_fails_when_fundamentals_coverage_is_low(offline):
    offline.setattr(pipeline, "fetch_fundamentals", lambda tk, *a, **k: {t: {} for t in tk[:5]})
    with pytest.raises(pipeline.DataQualityError, match="50%"):
        pipeline.load_dataset(Settings(strict=True))


def test_non_strict_does_not_raise_on_low_coverage(offline):
    offline.setattr(pipeline, "fetch_fundamentals", lambda tk, *a, **k: {t: {} for t in tk[:5]})
    offline.setattr(pipeline, "fetch_prices", lambda *a, **k: (_ for _ in ()).throw(StopIteration("reached prices")))
    with pytest.raises(StopIteration, match="reached prices"):
        pipeline.load_dataset(Settings(strict=False))
