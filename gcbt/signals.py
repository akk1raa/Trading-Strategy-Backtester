"""Turn a StrategySpec's rules into boolean signal series (evaluated at bar close)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ta
from .spec import Condition, Operand, Rules, SpecError, StrategySpec


class _Ctx:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cache: dict[str, pd.Series] = {}

    def operand(self, o: Operand) -> pd.Series:
        df = self.df
        if o.value is not None:
            return pd.Series(o.value, index=df.index)
        key = repr((o.ind, sorted(o.params.items()), o.field_))
        if key in self.cache:
            return self.cache[key]
        p = o.params
        src = df[p.get("source", "close")] if p.get("source", "close") in df else df["close"]
        if o.ind in ("close", "open", "high", "low"):
            out = df[o.ind]
        elif o.ind == "sma":
            out = ta.sma(src, int(p.get("period", 20)))
        elif o.ind == "ema":
            out = ta.ema(src, int(p.get("period", 20)))
        elif o.ind == "rsi":
            out = ta.rsi(src, int(p.get("period", 14)))
        elif o.ind == "atr":
            out = ta.atr(df, int(p.get("period", 14)))
        elif o.ind == "bbands":
            b = ta.bbands(src, int(p.get("period", 20)), float(p.get("std", 2.0)))
            out = b[o.field_ or "mid"]
        elif o.ind == "macd":
            m = ta.macd(src, int(p.get("fast", 12)), int(p.get("slow", 26)), int(p.get("signal", 9)))
            out = m[o.field_ or "line"]
        elif o.ind == "donchian":
            d = ta.donchian(df, int(p.get("period", 20)))
            out = d[o.field_ or "upper"]
        elif o.ind == "vwap":
            out = ta.vwap(df)
        else:  # pragma: no cover - spec validation prevents this
            raise SpecError(f"Unsupported indicator {o.ind}")
        self.cache[key] = out
        return out

    def condition(self, c: Condition) -> pd.Series:
        l, r = self.operand(c.left), self.operand(c.right)
        if c.op == ">":
            s = l > r
        elif c.op == "<":
            s = l < r
        elif c.op == ">=":
            s = l >= r
        elif c.op == "<=":
            s = l <= r
        elif c.op == "crosses_above":
            s = (l > r) & (l.shift(1) <= r.shift(1))
        elif c.op == "crosses_below":
            s = (l < r) & (l.shift(1) >= r.shift(1))
        else:  # pragma: no cover
            raise SpecError(c.op)
        return s.fillna(False)

    def rules(self, r: Rules) -> pd.Series:
        if r.empty():
            return pd.Series(False, index=self.df.index)
        out = pd.Series(True, index=self.df.index)
        for c in r.all_of:
            out &= self.condition(c)
        if r.any_of:
            anyc = pd.Series(False, index=self.df.index)
            for c in r.any_of:
                anyc |= self.condition(c)
            out &= anyc
        return out.astype(bool)


def build_signals(df: pd.DataFrame, spec: StrategySpec) -> dict[str, np.ndarray]:
    ctx = _Ctx(df)
    use_long = spec.direction in ("long", "both")
    use_short = spec.direction in ("short", "both")
    zeros = np.zeros(len(df), dtype=bool)
    atr_period = max(spec.stop.atr_period, spec.target.atr_period)
    return {
        "long_entry": ctx.rules(spec.long_entry).to_numpy() if use_long else zeros,
        "long_exit": ctx.rules(spec.long_exit).to_numpy() if use_long else zeros,
        "short_entry": ctx.rules(spec.short_entry).to_numpy() if use_short else zeros,
        "short_exit": ctx.rules(spec.short_exit).to_numpy() if use_short else zeros,
        "atr": ta.atr(df, atr_period).to_numpy(),
    }
