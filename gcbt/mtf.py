"""Run one strategy spec across 5m / 15m / 1h and report win rate for the last week and last month."""
from __future__ import annotations

import pandas as pd

from .data import load_tradingview_csv, load_yfinance, resample, synthetic
from .engine import CostConfig, run_backtest
from .spec import StrategySpec

TIMEFRAMES = ("5m", "15m", "1h")
WINDOWS = {"week": 7, "month": 30}


def load_frames(source: str, csv_path: str | None = None) -> dict[str, pd.DataFrame]:
    """Return {timeframe: OHLCV}. A 5m CSV is resampled up; yfinance is fetched per interval."""
    if source == "yfinance":
        return {tf: load_yfinance("GC=F", tf) for tf in TIMEFRAMES}
    if source == "csv":
        if not csv_path:
            raise ValueError("source 'csv' needs a path to a 5-minute TradingView export.")
        base = load_tradingview_csv(csv_path)
        return {"5m": base, "15m": resample(base, "15m"), "1h": resample(base, "1h")}
    if source == "synthetic":
        base = synthetic(n=12 * 24 * 45, tf="5m")  # ~45 days of 5m bars; demo only
        return {"5m": base, "15m": resample(base, "15m"), "1h": resample(base, "1h")}
    raise ValueError(f"Unknown source '{source}'")


def _window_stats(trades: pd.DataFrame, since: pd.Timestamp) -> dict:
    t = trades[trades.entry_time >= since] if not trades.empty else trades
    n = len(t)
    if n == 0:
        return {"trades": 0, "wins": 0, "win_rate": None, "net_pnl": 0.0, "profit_factor": None, "curve": []}
    wins = int((t.pnl > 0).sum())
    gl = -t.loc[t.pnl <= 0, "pnl"].sum()
    return {
        "trades": n, "wins": wins, "win_rate": wins / n,
        "net_pnl": float(t.pnl.sum()),
        "profit_factor": float(t.loc[t.pnl > 0, "pnl"].sum() / gl) if gl > 0 else None,
        "curve": [round(x, 2) for x in t.pnl.cumsum().tolist()],
    }


def run_multi(spec: StrategySpec, source: str = "yfinance", csv_path: str | None = None,
              cfg: CostConfig | None = None) -> dict:
    cfg = cfg or CostConfig()
    frames = load_frames(source, csv_path)
    out: dict = {}
    for tf in TIMEFRAMES:
        df = frames[tf]
        res = run_backtest(df, spec, cfg)  # indicators warm up on all available history
        end = df.index[-1]
        out[tf] = {
            "bars": int(len(df)), "data_start": str(df.index[0]), "data_end": str(end),
            **{name: _window_stats(res.trades, end - pd.Timedelta(days=days)) for name, days in WINDOWS.items()},
        }
    return {
        "strategy": {"name": spec.name, "direction": spec.direction, "confidence": spec.confidence,
                     "uncodifiable": spec.uncodifiable, "spec_timeframe": spec.timeframe},
        "source": source,
        "note": "Indicator periods are in BARS, so the same rules run on a different clock at each timeframe.",
        "results": out,
    }


def format_markdown(result: dict) -> str:
    """Compact table of win rate by timeframe and window (used by the CLI and the scheduled GitHub job)."""
    s = result["strategy"]
    lines = [f"## {s['name']}", f"direction: {s['direction']} | confidence: {s['confidence']} | data: {result['source']}", ""]
    if s["uncodifiable"]:
        lines += ["**Not backtested (could not be codified):**"] + [f"- {u}" for u in s["uncodifiable"]] + [""]
    lines += ["| Timeframe | Window | Win rate | Trades | Profit factor | Net P&L | Note |", "|---|---|---|---|---|---|---|"]
    for tf, r in result["results"].items():
        for w in WINDOWS:
            x = r[w]
            wr = "--" if x["win_rate"] is None else f"{x['win_rate']:.1%}"
            pf = "--" if x["profit_factor"] is None else f"{x['profit_factor']:.2f}"
            note = "LOW SAMPLE" if x["trades"] < 10 else ""
            lines.append(f"| {tf} | {w} | {wr} | {x['trades']} | {pf} | ${x['net_pnl']:,.0f} | {note} |")
    lines += ["", f"_{result['note']} Not financial advice._"]
    return "\n".join(lines)
