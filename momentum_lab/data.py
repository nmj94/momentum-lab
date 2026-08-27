"""data.py - Download market data for any ticker."""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Default to the repository-local data directory, but allow an environment
# override: in a wheel install ``Path(__file__).parent.parent`` points at the
# interpreter's site-packages parent, and we must not write market data there.
DATA_DIR = Path(os.environ.get("MOMENTUM_LAB_DATA_DIR", str(Path(__file__).parent.parent / "data")))

# Yahoo tickers are alphanumeric plus a small symbol set (BRK-B, RDS.A,
# ^GSPC, EURUSD=X).  Anything else (notably URL-significant characters such
# as ``?``, ``&``, ``#`` or ``/``) is rejected before it reaches the request
# URL built by yfinance.
_TICKER_RE = re.compile(r"^[A-Za-z0-9._^=-]+$")


def _load_cache(cache_path):
    try:
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
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


def _range_boundaries(start, end):
    """Business-day adjusted [start, end] boundaries for coverage checks."""
    start_ts = pd.Timestamp(start)
    # Yahoo's ``end`` bound is exclusive and market data is not published on
    # weekends.  Compare against the nearest business-day boundaries while
    # still slicing the user-facing result inclusively.
    start_boundary = start_ts + pd.offsets.BDay(1) if start_ts.weekday() >= 5 else start_ts
    now = pd.Timestamp.now().normalize()
    end_ts = pd.Timestamp(end) if end else now
    # A future ``end`` can never be covered by any download; cap it at today
    # instead of failing on the maximal available history.
    end_ts = min(end_ts, now)
    # Open-ended or near-current requests get one extra business day of
    # slack: providers emit a NaN placeholder row for the still-live
    # session, so the newest complete bar can lag "now" by two sessions
    # depending on timezone.  Explicitly historical ends stay strict.
    open_ended = end is None or pd.Timestamp(end).normalize() >= now - pd.Timedelta(days=2)
    slack = pd.offsets.BDay(2) if open_ended else pd.offsets.BDay(1)
    return start_boundary, end_ts - slack


def _cache_covers_range(df, start, end):
    """Return whether a cached frame fully covers the requested date range."""
    if df is None or df.empty:
        return False
    start_boundary, end_boundary = _range_boundaries(start, end)
    # Allow a small start gap for exchange holidays that are not represented
    # by pandas' generic business-day offset, but never accept a truncated end.
    tolerance = pd.Timedelta(days=7)
    if df.index.min() > start_boundary + tolerance:
        return False
    return df.index.max() >= end_boundary


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
        ValueError: If the ticker is invalid, the download fails and no
            cached copy exists, or the download contains no usable rows.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else None
    if end_ts is not None and end_ts < start_ts:
        raise ValueError("end must be on or after start")

    ticker = str(ticker)
    if not _TICKER_RE.match(ticker):
        raise ValueError(f"Invalid ticker {ticker!r}: only letters, digits and '.', '_', '^', '=', '-' are allowed.")

    DATA_DIR.mkdir(exist_ok=True)
    safe_ticker = "".join(char if char.isalnum() or char in "._-" else "_" for char in ticker)
    cache_path = DATA_DIR / f"{safe_ticker}_daily.csv"

    cached = _load_cache(cache_path) if use_cache and cache_path.exists() else None
    if cached is not None and _cache_covers_range(cached, start, end):
        return _slice_range(cached, start, end)
    # A partial cache must not be returned silently.  Refresh the requested
    # window and merge it with the existing cache after a successful download.

    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as e:
        # Temporary failure (e.g. Yahoo rate limit). Fall back to cached data.
        if cached is not None and _cache_covers_range(cached, start, end):
            warnings.warn(f"Download failed ({e}); falling back to cached data.", RuntimeWarning)
            return _slice_range(cached, start, end)
        raise

    if df is None or df.empty:
        if cached is not None and _cache_covers_range(cached, start, end):
            warnings.warn("Download returned no data; falling back to cached data.", RuntimeWarning)
            return _slice_range(cached, start, end)
        raise ValueError(
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

    # The provider may return a narrower range than requested (for example
    # for a newly listed or delisted symbol).  A truncated END is always a
    # hard error: silently serving stale history would poison the research
    # engine.  A late START, however, is usually legitimate - many assets
    # are listed after the default 2004-01-01 start (GLD IPO 2004-11-18,
    # BTC-USD 2014) and we cannot distinguish that from a provider gap
    # locally, so downgrade it to a warning.  Validate BEFORE persisting so
    # a rejected window never contaminates the cache.
    start_boundary, end_boundary = _range_boundaries(start, end)
    if df.index.max() < end_boundary:
        raise ValueError(f"Downloaded data for '{ticker}' does not cover the requested range.")
    if df.index.min() > start_boundary + pd.Timedelta(days=7):
        warnings.warn(
            f"Data for '{ticker}' starts at {df.index.min().date()}, later than the requested "
            f"start {pd.Timestamp(start).date()}; treating it as the earliest available history.",
            RuntimeWarning,
        )

    if use_cache:
        df.to_csv(cache_path)

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


def prepare_data(ticker="GLD", start="2004-01-01", end=None, use_cache=True, annualization=252.0):
    """Download data + compute features, return unified data dict.

    Args:
        ticker: Yahoo Finance ticker.
        start: Start date.
        end: End date.
        use_cache: If True, reuse the local CSV cache (default True).
        annualization: Return periods per year. Strategies read it from the
            data dict so volatility targeting matches the evaluation horizon
            (252 for trading days, 365 for continuously traded assets).

    Returns:
        (data_dict, df) where data_dict has keys for strategy consumption
        and df is the raw OHLCV DataFrame.
    """
    df = download_data(ticker, start=start, end=end, use_cache=use_cache)
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
    }
    return data, df
