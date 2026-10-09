"""Event-driven backtest engine for GC / MGC.

Conventions (these are what keep results honest):
  * Signals are computed on bar i's close and filled at bar i+1's OPEN. No lookahead.
  * Stops/targets are checked intrabar using high/low. If a bar touches both, the STOP wins.
  * Gaps through a stop fill at the (worse) open, not at the stop price.
  * Market fills (entries, signal exits, stops) pay slippage; resting target fills do not.
  * Commission is charged per contract per side.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import build_signals
from .spec import StrategySpec

# Contract specs: (point value in $, tick size)
CONTRACTS = {
    "GC": (100.0, 0.10),   # 100 oz, $10 per 0.10 tick
    "MGC": (10.0, 0.10),   # 10 oz,  $1  per 0.10 tick
}


@dataclass
class CostConfig:
    symbol: str = "GC"
    commission_per_side: float = 2.50   # $ per contract per side (all-in; adjust to your broker)
    slippage_ticks: float = 1.0         # ticks per market fill
    initial_capital: float = 50_000.0

    @property
    def multiplier(self) -> float:
        return CONTRACTS[self.symbol][0]

    @property
    def tick(self) -> float:
        return CONTRACTS[self.symbol][1]


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    spec: StrategySpec
    cfg: CostConfig


def _in_session(hour: int, window: tuple[int, int] | None) -> bool:
    if window is None:
        return True
    s, e = window
    return (s <= hour < e) if s < e else (hour >= s or hour < e)  # supports overnight windows


def _level(kind: str, value: float, price: float, atr_val: float) -> float | None:
    """Distance (in price points) for a stop/target, or None if disabled/unavailable."""
    if kind == "none" or value <= 0:
        return None
    if kind == "fixed_points":
        return value
    if kind == "percent":
        return price * value / 100.0
    if kind == "atr_multiple":
        return None if np.isnan(atr_val) else value * atr_val
    return None


def run_backtest(df: pd.DataFrame, spec: StrategySpec, cfg: CostConfig | None = None) -> BacktestResult:
    cfg = cfg or CostConfig()
    sig = build_signals(df, spec)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    hours = df.index.hour.to_numpy()
    n = len(df)
    mult, tick = cfg.multiplier, cfg.tick
    slip = cfg.slippage_ticks * tick
    q = spec.contracts
    fee = cfg.commission_per_side * q

    pos = 0
    entry = stop = tgt = np.nan
    entry_i = 0
    cash = 0.0
    equity = np.zeros(n)
    trades: list[dict] = []

    def close_trade(i: int, px: float, reason: str) -> None:
        nonlocal pos, cash
        gross = pos * (px - entry) * mult * q
        cash += gross - fee  # exit-side commission (entry-side was charged at entry)
        trades.append({
            "entry_time": df.index[entry_i], "exit_time": df.index[i],
            "side": "long" if pos == 1 else "short",
            "entry": entry, "exit": px, "contracts": q,
            "pnl": gross - 2 * fee, "reason": reason, "bars": i - entry_i,
        })
        pos = 0

    for i in range(n):
        if i > 0:
            # 1) Signal-based exits fill at this bar's open.
            if pos == 1 and (sig["long_exit"][i - 1] or sig["short_entry"][i - 1]):
                close_trade(i, o[i] - slip, "signal")
            elif pos == -1 and (sig["short_exit"][i - 1] or sig["long_entry"][i - 1]):
                close_trade(i, o[i] + slip, "signal")

            # 2) Entries fill at this bar's open.
            if pos == 0 and _in_session(hours[i], spec.session_utc):
                side = 0
                if sig["long_entry"][i - 1] and not sig["long_exit"][i - 1]:
                    side = 1
                elif sig["short_entry"][i - 1] and not sig["short_exit"][i - 1]:
                    side = -1
                if side != 0:
                    atr_prev = sig["atr"][i - 1]
                    sd = _level(spec.stop.kind, spec.stop.value, o[i], atr_prev)
                    td = _level(spec.target.kind, spec.target.value, o[i], atr_prev)
                    # If a stop/target was requested but can't be computed (ATR warm-up), skip the trade.
                    if (spec.stop.kind != "none" and sd is None) or (spec.target.kind != "none" and td is None):
                        side = 0
                if side != 0:
                    pos, entry_i = side, i
                    entry = o[i] + slip if side == 1 else o[i] - slip
                    cash -= fee
                    stop = entry - side * sd if sd else np.nan
                    tgt = entry + side * td if td else np.nan

        # 3) Intrabar stop / target (including the entry bar, after the open).
        if pos == 1:
            if not np.isnan(stop) and l[i] <= stop:
                close_trade(i, min(stop, o[i]) - slip, "stop")
            elif not np.isnan(tgt) and h[i] >= tgt:
                close_trade(i, max(tgt, o[i]) if i > entry_i else tgt, "target")
        elif pos == -1:
            if not np.isnan(stop) and h[i] >= stop:
                close_trade(i, max(stop, o[i]) + slip, "stop")
            elif not np.isnan(tgt) and l[i] <= tgt:
                close_trade(i, min(tgt, o[i]) if i > entry_i else tgt, "target")

        # 4) Mark to market at the close.
        unreal = pos * (c[i] - entry) * mult * q if pos != 0 else 0.0
        equity[i] = cfg.initial_capital + cash + unreal

    if pos != 0:  # flatten at the final close so the last trade is counted
        close_trade(n - 1, c[-1], "end_of_data")
        equity[-1] = cfg.initial_capital + cash

    return BacktestResult(pd.DataFrame(trades), pd.Series(equity, index=df.index, name="equity"), spec, cfg)
