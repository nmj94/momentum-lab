"""data.py - Download market data for any ticker."""

import json
import os
import re
import time
import warnings
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import yfinance as yf
from platformdirs import user_cache_dir

from .artifacts import atomic_text_output

# Cache outside the package directory so wheel installs never try to write into
# site-packages.  The environment override remains useful for CI and containers.
DATA_DIR = Path(os.environ.get("MOMENTUM_LAB_DATA_DIR", user_cache_dir("momentum-lab")))
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_BACKOFF_SECONDS = 0.5
CACHE_SCHEMA_VERSION = 1

# Yahoo tickers are alphanumeric plus a small symbol set (BRK-B, RDS.A,
# ^GSPC, EURUSD=X).  Anything else (notably URL-significant characters such
# as ``?``, ``&``, ``#`` or ``/``) is rejected before it reaches the request
# URL built by yfinance.
_TICKER_RE = re.compile(r"[A-Za-z0-9._^=-]{1,64}")
_CONTINUOUS_QUOTES = ("-USD", "-EUR", "-GBP", "-JPY", "-AUD", "-CAD", "-CHF", "-BTC", "-ETH")


class MarketDataUnavailableError(RuntimeError):
    """The remote provider could not supply data after bounded retries."""


def is_continuously_traded(ticker: str) -> bool:
    """Return whether a Yahoo ticker is a continuously traded crypto pair."""
    return str(ticker).upper().endswith(_CONTINUOUS_QUOTES)


def infer_annualization(ticker: str) -> float:
    """Infer daily return periods per year from the asset's trading calendar."""
    return 365.0 if is_continuously_traded(ticker) else 252.0


def _load_cache(cache_path):
    try:
        return pd.read_csv(cache_path, index_col=0, parse_dates=True, float_precision="round_trip")
    except (OSError, ValueError) as e:
        # A corrupt or truncated cache file must not permanently brick every
        # future download; treat it as a cache miss and re-download.
        warnings.warn(f"Ignoring unreadable cache file {cache_path}: {e}", RuntimeWarning)
        return None


def _slice_range(df, start, end):
    """Slice a DataFrame's DatetimeIndex to [start, end] inclusive."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else None
    mask = df.index >= start_ts
    if end_ts is not None:
        mask &= df.index <= end_ts
    return df.loc[mask]


def _range_boundaries(start, end, continuous=False):
    """Calendar-adjusted inclusive boundaries for cache coverage checks."""
    start_ts = pd.Timestamp(start)
    start_boundary = start_ts if continuous else pd.offsets.BDay().rollforward(start_ts)
    now = pd.Timestamp.now().normalize()
    end_ts = pd.Timestamp(end) if end else now
    # A future ``end`` can never be covered by any download; cap it at today
    # instead of failing on the maximal available history.
    end_ts = min(end_ts, now)
    # Open-ended or near-current requests get two sessions of
    # slack: providers emit a NaN placeholder row for the still-live
    # session, so the newest complete bar can lag "now" by two sessions
    # depending on timezone. Explicitly historical ends stay strict and are
    # inclusive, matching this module's public contract.
    open_ended = end is None or pd.Timestamp(end).normalize() >= now - pd.Timedelta(days=2)
    if open_ended:
        slack = pd.Timedelta(days=2) if continuous else pd.offsets.BDay(2)
        end_boundary = end_ts - slack
    else:
        end_boundary = end_ts if continuous else pd.offsets.BDay().rollback(end_ts)
    return start_boundary, end_boundary


def _cache_covers_range(df, start, end, continuous=False, earliest_available=None):
    """Return whether a cached frame fully covers the requested date range."""
    if df is None or df.empty:
        return False
    start_boundary, end_boundary = _range_boundaries(start, end, continuous=continuous)
    # Allow a small start gap for exchange holidays that are not represented
    # by pandas' generic business-day offset, but never accept a truncated end.
    tolerance = pd.Timedelta(days=7)
    if df.index.min() > start_boundary + tolerance:
        # A sidecar written after a successful provider response records that
        # the symbol genuinely starts later than the requested history.  This
        # lets late-listed assets reuse their cache without relaxing coverage
        # checks for legacy or accidentally truncated files.
        if earliest_available is None:
            return False
        earliest = pd.Timestamp(earliest_available)
        if start_boundary >= earliest or abs(df.index.min() - earliest) > tolerance:
            return False
    return df.index.max() >= end_boundary


def _metadata_path(cache_path):
    return cache_path.with_suffix(".meta.json")


def _load_cache_metadata(cache_path, ticker):
    path = _metadata_path(cache_path)
    if not path.exists():
        return {}
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(metadata, dict)
            or type(metadata.get("schema_version")) is not int
            or metadata["schema_version"] != CACHE_SCHEMA_VERSION
            or metadata.get("ticker") != ticker
        ):
            raise ValueError("schema or ticker identity mismatch")
        for name in ("earliest_available", "latest_available", "requested_start"):
            value = metadata.get(name)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(f"invalid {name}")
                _daily_bound(value, name)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        warnings.warn(f"Ignoring invalid cache metadata {path}: {exc}", RuntimeWarning)
        return None  # Invalid identity must invalidate prices too, not merely coverage hints.
    return metadata


def _write_cache_atomic(df, cache_path):
    """Write a cache snapshot atomically so interruption cannot truncate it."""
    with atomic_text_output(cache_path, newline="") as handle:
        df.to_csv(handle)


def _write_cache_metadata_atomic(metadata, cache_path):
    path = _metadata_path(cache_path)
    with atomic_text_output(path) as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))


def _daily_bound(value, name):
    try:
        if not isinstance(value, (str, pd.Timestamp)):
            raise TypeError("expected a daily date")
        bound = pd.Timestamp(value)
        if pd.isna(bound) or bound.tzinfo is not None or bound != bound.normalize():
            raise ValueError("expected a timezone-free daily date")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a valid timezone-free daily date") from exc
    return bound


def _validate_ohlcv(df, ticker):
    """Enforce the provider contract before data reaches any strategy."""
    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Download for '{ticker}' is missing required OHLC column(s): {', '.join(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Download for '{ticker}' does not have a DatetimeIndex.")
    if df.index.hasnans or not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError(f"Download for '{ticker}' has an unsorted or duplicate date index.")
    if df.index.tz is not None or not df.index.equals(df.index.normalize()):
        raise ValueError(f"Download for '{ticker}' requires timezone-free daily session dates.")
    if not df.columns.is_unique:
        raise ValueError(f"Download for '{ticker}' contains duplicate columns.")
    prices = df[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(f"Download for '{ticker}' contains non-finite or non-positive OHLC prices.")
    if (df["high"] < df[["open", "low", "close"]].max(axis=1)).any() or (
        df["low"] > df[["open", "high", "close"]].min(axis=1)
    ).any():
        raise ValueError(f"Download for '{ticker}' contains inconsistent high/low prices.")
    if "volume" in df:
        volume = df["volume"].dropna().to_numpy(dtype=float)
        if not np.isfinite(volume).all() or (volume < 0).any():
            raise ValueError(f"Download for '{ticker}' contains non-finite or negative volume.")


def download_data(ticker="GLD", start="2004-01-01", end=None, use_cache=True):
    """Download OHLCV data for any ticker from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. "GLD", "SPY", "BTC-USD").
        start: Start date string (YYYY-MM-DD).
        end: End date string. None = today.
        use_cache: If True, cache to local CSV for faster re-runs.

    Returns:
        pd.DataFrame with columns: open, high, low, close, volume.

    Raises:
        ValueError: If the ticker/range is invalid or data violates the OHLC contract.
        MarketDataUnavailableError: If the provider fails after bounded retries
            and no range-complete cache exists.
    """
    start_ts = _daily_bound(start, "start")
    end_ts = _daily_bound(end, "end") if end is not None else None
    if end_ts is not None and end_ts < start_ts:
        raise ValueError("end must be on or after start")

    if not isinstance(ticker, str) or not _TICKER_RE.fullmatch(ticker):
        raise ValueError(f"Invalid ticker {ticker!r}: only letters, digits and '.', '_', '^', '=', '-' are allowed.")
    ticker = ticker.upper()

    continuous = is_continuously_traded(ticker)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Encode underscore too: old lossy cache names used it for both '^' and
    # '='. No ambiguous legacy filename is reused for ANY of those tickers.
    safe_ticker = quote(ticker, safe=".-").replace("_", "%5F")
    cache_path = DATA_DIR / f"{safe_ticker}_daily.csv"

    cached = _load_cache(cache_path) if use_cache and cache_path.exists() else None
    if cached is not None:
        try:
            _validate_ohlcv(cached, ticker)
        except (TypeError, ValueError) as exc:
            warnings.warn(f"Ignoring invalid cache file {cache_path}: {exc}", RuntimeWarning)
            cached = None
    cache_metadata = _load_cache_metadata(cache_path, ticker) if cached is not None else {}
    if cache_metadata is None:
        cached, cache_metadata = None, {}
    earliest_available = cache_metadata.get("earliest_available")
    if cached is not None and _cache_covers_range(
        cached,
        start,
        end,
        continuous=continuous,
        earliest_available=earliest_available,
    ):
        return _slice_range(cached, start, end)
    # A partial cache must not be returned silently.  Refresh the requested
    # window and merge it with the existing cache after a successful download.

    # yfinance defines ``end`` as exclusive while this API promises an
    # inclusive [start, end] range. Ask the provider for one extra calendar day.
    provider_end = str((end_ts + pd.Timedelta(days=1)).date()) if end_ts is not None else None
    df = None
    last_error = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            candidate = yf.download(ticker, start=start, end=provider_end, auto_adjust=True, progress=False)
            if candidate is not None and not candidate.empty:
                df = candidate
                break
        except Exception as exc:
            last_error = exc
        if attempt + 1 < DOWNLOAD_ATTEMPTS:
            time.sleep(DOWNLOAD_BACKOFF_SECONDS * (2**attempt))

    if df is None and last_error is not None:
        # Temporary failure (e.g. Yahoo rate limit). Fall back to cached data.
        if cached is not None and _cache_covers_range(
            cached, start, end, continuous=continuous, earliest_available=earliest_available
        ):
            warnings.warn(f"Download failed ({last_error}); falling back to cached data.", RuntimeWarning)
            return _slice_range(cached, start, end)
        raise MarketDataUnavailableError(f"Market-data provider failed for '{ticker}': {last_error}") from last_error

    if df is None or df.empty:
        if cached is not None and _cache_covers_range(
            cached, start, end, continuous=continuous, earliest_available=earliest_available
        ):
            warnings.warn("Download returned no data; falling back to cached data.", RuntimeWarning)
            return _slice_range(cached, start, end)
        raise MarketDataUnavailableError(
            f"Download failed for '{ticker}'. "
            f"Possible causes: invalid ticker, delisted, or network error. "
            f"Try searching at https://finance.yahoo.com/lookup"
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        # Yahoo daily labels use the exchange's local session date. Drop the
        # timezone without converting to UTC (which could shift the date).
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # Only price columns are mandatory: volume is NaN for some indices and
    # must not punch holes in the price series.  A frame left empty after
    # dropping price NaNs is a hard failure, never a silent empty result
    # (previously it slid past the coverage checks and poisoned the cache).
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    df = df.dropna(subset=price_cols)
    if df.empty or "close" not in df.columns:
        raise ValueError(f"Download for '{ticker}' returned no usable data.")

    if use_cache and cached is not None:
        # Keep previously downloaded history instead of overwriting it
        # with the newly requested window.
        df = pd.concat([cached, df]).sort_index()
        df = df[~df.index.duplicated(keep="last")]

    _validate_ohlcv(df, ticker)

    # The provider may return a narrower range than requested (for example
    # for a newly listed or delisted symbol).  A truncated END is always a
    # hard error: silently serving stale history would poison the research
    # engine.  A late START, however, is usually legitimate - many assets
    # are listed after the default 2004-01-01 start (GLD IPO 2004-11-18,
    # BTC-USD 2014) and we cannot distinguish that from a provider gap
    # locally, so downgrade it to a warning.  Validate BEFORE persisting so
    # a rejected window never contaminates the cache.
    start_boundary, end_boundary = _range_boundaries(start, end, continuous=continuous)
    if df.index.max() < end_boundary:
        raise ValueError(f"Downloaded data for '{ticker}' does not cover the requested range.")
    late_start = df.index.min() > start_boundary + pd.Timedelta(days=7)
    if late_start:
        warnings.warn(
            f"Data for '{ticker}' starts at {df.index.min().date()}, later than the requested "
            f"start {pd.Timestamp(start).date()}; treating it as the earliest available history.",
            RuntimeWarning,
        )

    if use_cache:
        confirmed_earliest = None
        if late_start:
            confirmed_earliest = str(df.index.min())
        elif earliest_available is not None and pd.Timestamp(earliest_available) == df.index.min():
            confirmed_earliest = str(pd.Timestamp(earliest_available))
        _write_cache_atomic(df, cache_path)
        _write_cache_metadata_atomic(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "ticker": ticker,
                "earliest_available": confirmed_earliest,
                "requested_start": str(start_ts),
                "latest_available": str(df.index.max()),
            },
            cache_path,
        )

    return _slice_range(df, start, end)


def compute_features(df, annualization=252.0):
    """Compute technical indicator features for ML strategies.

    Args:
        df: DataFrame with at least a 'close' column.
        annualization: Return periods per year, used to annualize the
            volatility features (252 for trading days, 365 for crypto).

    Returns:
        pd.DataFrame of features (same index as input).
    """
    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)
    # NaN volume (indices, sparse feeds) must be zero-filled here: left as
    # NaN it poisons vol_ratio below, punching NaN holes that silently drop
    # rows from ML training windows - or, for an all-NaN column, drop EVERY
    # row and flatten all ML positions without any error being raised.
    volume = df.get("volume", pd.Series(1, index=df.index)).fillna(0)

    feats = pd.DataFrame(index=df.index)

    # Returns at various lookbacks
    for lb in [1, 3, 5, 10, 21, 42, 63, 126, 252]:
        feats[f"ret_{lb}"] = close.pct_change(lb)

    # MA distances
    for lb in [5, 10, 20, 50, 100, 200]:
        ma = close.rolling(lb).mean()
        feats[f"ma_dist_{lb}"] = (close - ma) / ma

    # RSI
    delta = close.diff()
    for period in [7, 14, 21, 28]:
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        feats[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # MACD
    for fast, slow in [(8, 21), (12, 26), (16, 35)]:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        for sig in [5, 9, 12]:
            signal = macd.ewm(span=sig, adjust=False).mean()
            feats[f"macd_{fast}_{slow}_{sig}"] = macd - signal

    # Volatility
    for lb in [10, 21, 42, 63]:
        feats[f"vol_{lb}"] = close.pct_change().rolling(lb).std() * np.sqrt(annualization)

    # Bollinger band position
    for lb in [10, 20, 50]:
        for nstd in [1.5, 2.0, 2.5]:
            ma = close.rolling(lb).mean()
            std = close.rolling(lb).std()
            upper = ma + nstd * std
            lower = ma - nstd * std
            feats[f"bb_pos_{lb}_{nstd}"] = (close - lower) / (upper - lower + 1e-10)

    # Rate of change
    for lb in [5, 10, 21, 42, 63, 126, 252]:
        feats[f"roc_{lb}"] = (close / close.shift(lb) - 1) * 100

    # Volume ratio (epsilon: zero-volume stretches must not produce NaN
    # features that silently drop rows from ML training windows)
    for lb in [5, 10, 21]:
        feats[f"vol_ratio_{lb}"] = volume / (volume.rolling(lb).mean() + 1e-10)

    # Acceleration
    feats["acceleration"] = close.pct_change(5) - close.pct_change(10)

    # High-low range
    for lb in [10, 21, 55]:
        feats[f"range_{lb}"] = (high.rolling(lb).max() - low.rolling(lb).min()) / close

    return feats


def prepare_data(ticker="GLD", start="2004-01-01", end=None, use_cache=True, annualization=None, dataset=None):
    """Load market data + compute features, return unified data dict.

    Args:
        ticker: Yahoo Finance ticker.
        start: Start date.
        end: End date.
        use_cache: If True, reuse the local CSV cache (default True).
        annualization: Return periods per year. ``None`` infers 365 for common
            Yahoo crypto pairs and 252 otherwise. Strategies read the resolved
            value from the data dict so volatility targeting matches evaluation.
        dataset: Optional offline manifest path. No downloads or Yahoo caches
            are used; annualization must match its declared daily convention.
            ``end=None`` means the snapshot's last date, not today.

    Returns:
        (data_dict, df) where data_dict has keys for strategy consumption and
        ``data_provenance`` declarations, and df is the raw OHLCV DataFrame.
    """
    if dataset is not None:
        from .datasets import load_dataset, resolve_dataset_annualization

        df, provenance = load_dataset(dataset, ticker=ticker, start=start, end=end)
        annualization = resolve_dataset_annualization(provenance, annualization)
    else:
        annualization = infer_annualization(ticker) if annualization is None else annualization
        df = download_data(ticker, start=start, end=end, use_cache=use_cache)
        provenance = {
            "provider": "yahoo",
            "source": "Yahoo Finance via yfinance",
            "license": "Provider terms apply; usage rights are not verified by Momentum Lab",
            "ticker": str(ticker).upper(),
            "frequency": "1d",
            "price_adjustment": "yfinance_auto_adjust",
            "calendar": "continuous" if is_continuously_traded(ticker) else "exchange",
            "annualization": float(annualization),
        }
    feats = compute_features(df, annualization=annualization)

    data = {
        "close": df["close"],
        "high": df.get("high", df["close"]),
        "low": df.get("low", df["close"]),
        "open": df.get("open", df["close"]),
        "volume": df.get("volume", pd.Series(1, index=df.index)),
        "features": feats,
        "ticker": ticker,
        "annualization": annualization,
        "data_provenance": provenance,
    }
    return data, df
