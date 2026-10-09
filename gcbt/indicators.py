"""Tested indicator library. All functions are causal (use only past/current bar)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, period: int = 20) -> pd.Series:
    return s.rolling(period).mean()


def ema(s: pd.Series, period: int = 20) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bbands(s: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = s.rolling(period).mean()
    sd = s.rolling(period).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + std * sd, "lower": mid - std * sd})


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"line": line, "signal": sig, "hist": line - sig})


def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    # shift(1): channel of the PRIOR n bars, so a breakout is detectable on the current bar.
    return pd.DataFrame({
        "upper": df["high"].rolling(period).max().shift(1),
        "lower": df["low"].rolling(period).min().shift(1),
    })


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP, reset each UTC day."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, np.nan).fillna(1.0)
    day = df.index.normalize()
    return (tp * vol).groupby(day).cumsum() / vol.groupby(day).cumsum()
