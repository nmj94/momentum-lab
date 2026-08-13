"""data.py - Download market data for any ticker."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent.parent / "data"


def download_data(ticker="GLD", start="2004-01-01", end=None, use_cache=True):
    """Download OHLCV data for any ticker from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. "GLD", "SPY", "BTC-USD").
        start: Start date string (YYYY-MM-DD).
        end: End date string. None = today.
        use_cache: If True, cache to local CSV for faster re-runs.

    Returns:
        pd.DataFrame with columns: open, high, low, close, volume.
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache_path = DATA_DIR / f"{ticker.replace('^', '_')}_daily.csv"

    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df is None or df.empty:
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
    df = df.dropna()

    if use_cache:
        df.to_csv(cache_path)

    return df


def compute_features(df):
    """Compute technical indicator features for ML strategies.

    Args:
        df: DataFrame with at least a 'close' column.

    Returns:
        pd.DataFrame of features (same index as input).
    """
    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)
    volume = df.get("volume", pd.Series(1, index=df.index))

    feats = pd.DataFrame(index=df.index)

    # Returns at various lookbacks
    for lb in [1, 3, 5, 10, 21, 42, 63, 126, 252]:
        feats[f"ret_{lb}"] = close.pct_change(lb)

    # MA distances
    for lb in [5, 10, 20, 50, 100, 200]:
        ma = close.rolling(lb).mean()
        feats[f"ma_dist_{lb}"] = (close - ma) / ma

    # RSI
    for period in [7, 14, 21, 28]:
        delta = close.diff()
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
        feats[f"vol_{lb}"] = close.pct_change().rolling(lb).std() * np.sqrt(252)

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

    # Volume ratio
    for lb in [5, 10, 21]:
        feats[f"vol_ratio_{lb}"] = volume / volume.rolling(lb).mean()

    # Acceleration
    feats["acceleration"] = close.pct_change(5) - close.pct_change(10)

    # High-low range
    for lb in [10, 21, 55]:
        feats[f"range_{lb}"] = (high.rolling(lb).max() - low.rolling(lb).min()) / close

    return feats


def prepare_data(ticker="GLD", start="2004-01-01", end=None):
    """Download data + compute features, return unified data dict.

    Args:
        ticker: Yahoo Finance ticker.
        start: Start date.
        end: End date.

    Returns:
        (data_dict, df) where data_dict has keys for strategy consumption
        and df is the raw OHLCV DataFrame.
    """
    df = download_data(ticker, start=start, end=end)
    feats = compute_features(df)

    data = {
        "close": df["close"],
        "high": df.get("high", df["close"]),
        "low": df.get("low", df["close"]),
        "open": df.get("open", df["close"]),
        "volume": df.get("volume", pd.Series(1, index=df.index)),
        "features": feats,
        "ticker": ticker,
    }
    return data, df
