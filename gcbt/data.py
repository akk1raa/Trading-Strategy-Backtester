"""GC data loading. Sources: TradingView CSV export, Databento-style CSV, yfinance (prototyping), synthetic (tests)."""
from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]

# yfinance intraday history limits (days). Anything finer than 1d is heavily capped.
YF_LIMITS = {"1m": 6, "5m": 58, "15m": 58, "30m": 58, "1h": 728, "1d": 36500}
TF_ALIASES = {"1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "60m": "1h", "1hr": "1h", "d": "1d", "1D": "1d"}


def norm_tf(tf: str) -> str:
    return TF_ALIASES.get(tf, tf)


def _pandas_rule(tf: str) -> str:
    tf = norm_tf(tf)
    unit = {"m": "min", "h": "h", "d": "D"}[tf[-1]]
    return f"{tf[:-1]}{unit}"


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    out = df.resample(_pandas_rule(tf)).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open", "close"])


def load_tradingview_csv(path: str) -> pd.DataFrame:
    """TradingView 'Export chart data' CSV: time, open, high, low, close, (Volume ...). Time is ISO or unix seconds."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    tcol = next(c for c in ("time", "date", "datetime", "timestamp", "ts_event") if c in raw.columns)
    t = raw[tcol]
    if pd.api.types.is_numeric_dtype(t):
        # TradingView uses unix seconds; Databento/brokers often use ms, us or ns. Detect by magnitude.
        m = float(t.abs().max())
        unit = "s" if m < 1e11 else "ms" if m < 1e14 else "us" if m < 1e17 else "ns"
        idx = pd.to_datetime(t, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(t, utc=True)
    df = raw.rename(columns={"vol": "volume"}).assign(_t=idx).set_index("_t").rename_axis("time")
    for col in OHLCV:
        if col not in df:
            df[col] = 0.0
    return df[OHLCV].sort_index().astype(float).pipe(lambda d: d[~d.index.duplicated()])


def load_yfinance(symbol: str = "GC=F", interval: str = "1h", start: str | None = None, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    interval = norm_tf(interval)
    if interval not in YF_LIMITS:
        raise ValueError(f"yfinance interval {interval!r} unsupported. Use one of {list(YF_LIMITS)}.")
    kw: dict = {"interval": interval, "auto_adjust": False, "progress": False}
    if start or end:
        kw.update(start=start, end=end)
    elif interval == "1d":
        kw["period"] = "max"
    else:
        # yfinance 'period' only accepts fixed strings (1mo, 1y, ...), so ask for an explicit window
        # just inside Yahoo's per-interval history limit instead of an arbitrary "59d".
        now = pd.Timestamp.now(tz="UTC")
        kw.update(start=(now - pd.Timedelta(days=YF_LIMITS[interval])).strftime("%Y-%m-%d"),
                  end=(now + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    raw = yf.download(symbol, **kw)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data (rate-limited, offline, or interval/date window out of range).")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]
    df = raw[OHLCV].astype(float)
    df.index = pd.DatetimeIndex(df.index).tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df.rename_axis("time").dropna()


def synthetic(n: int = 5000, tf: str = "1h", seed: int = 7, start_price: float = 2000.0) -> pd.DataFrame:
    """Random-walk OHLCV with mild trend regimes. For tests/demos only — NOT real gold."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 0.02, n // 500 + 1), 500)[:n]
    ret = drift + rng.normal(0, 0.35, n)
    close = start_price + np.cumsum(ret)
    open_ = np.concatenate([[start_price], close[:-1]]) + rng.normal(0, 0.05, n)
    spread = np.abs(rng.normal(0.4, 0.2, n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range("2022-01-03", periods=n, freq=_pandas_rule(tf), tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.integers(100, 5000, n).astype(float)}, index=idx).rename_axis("time")
