"""Performance metrics, in/out-of-sample split, and per-year (regime) breakdown."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult


def _trade_stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0, "net_pnl": 0.0, "win_rate": np.nan, "profit_factor": np.nan,
                "expectancy": np.nan, "avg_win": np.nan, "avg_loss": np.nan}
    wins, losses = t.loc[t.pnl > 0, "pnl"], t.loc[t.pnl <= 0, "pnl"]
    gl = -losses.sum()
    return {
        "trades": int(len(t)),
        "net_pnl": float(t.pnl.sum()),
        "win_rate": float((t.pnl > 0).mean()),
        "profit_factor": float(wins.sum() / gl) if gl > 0 else float("inf"),
        "expectancy": float(t.pnl.mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
    }


def _curve_stats(eq: pd.Series, capital: float) -> dict:
    daily = eq.resample("1D").last().dropna()
    rets = daily.diff().dropna() / capital
    sd = rets.std(ddof=1)
    dsd = rets[rets < 0].std(ddof=1)
    peak = eq.cummax()
    dd = eq - peak
    return {
        "sharpe": float(rets.mean() / sd * np.sqrt(252)) if sd and sd > 0 else float("nan"),
        "sortino": float(rets.mean() / dsd * np.sqrt(252)) if dsd and dsd > 0 else float("nan"),
        "max_drawdown_usd": float(dd.min()),
        "max_drawdown_pct": float((dd / peak).min() * 100),
    }


def summarize(res: BacktestResult, oos_fraction: float = 0.3) -> dict:
    t, eq, cap = res.trades, res.equity, res.cfg.initial_capital
    out = {"overall": {**_trade_stats(t), **_curve_stats(eq, cap),
                       "start": str(eq.index[0]), "end": str(eq.index[-1]), "bars": int(len(eq))}}
    if not t.empty:
        cut = eq.index[int(len(eq) * (1 - oos_fraction))]
        out["in_sample"] = _trade_stats(t[t.entry_time < cut])
        out["out_of_sample"] = _trade_stats(t[t.entry_time >= cut])
        out["oos_cutoff"] = str(cut)
        by_year = t.groupby(t.exit_time.dt.year)["pnl"].agg(["count", "sum"])
        out["by_year"] = {int(y): {"trades": int(r["count"]), "net_pnl": float(r["sum"])} for y, r in by_year.iterrows()}
        out["by_side"] = {s: _trade_stats(g) for s, g in t.groupby("side")}
        out["exit_reasons"] = t.reason.value_counts().to_dict()
    return out


def format_report(spec, summary: dict) -> str:
    o = summary["overall"]
    L = [f"# {spec.name}", f"Timeframe {spec.timeframe} | direction {spec.direction} | confidence {spec.confidence}", ""]
    if spec.uncodifiable:
        L += ["**Could NOT be codified (backtest ignores these):**"] + [f"- {u}" for u in spec.uncodifiable] + [""]
    L += [f"Period: {o['start'][:10]} -> {o['end'][:10]}  ({o['bars']} bars)", "",
          f"Trades {o['trades']} | Net P&L ${o['net_pnl']:,.0f} | Win rate {o['win_rate']:.1%} | "
          f"PF {o['profit_factor']:.2f} | Expectancy ${o['expectancy']:,.0f}/trade",
          f"Sharpe {o['sharpe']:.2f} | Sortino {o['sortino']:.2f} | Max DD ${o['max_drawdown_usd']:,.0f} ({o['max_drawdown_pct']:.1f}%)", ""]
    for k, label in (("in_sample", "In-sample"), ("out_of_sample", "Out-of-sample")):
        if k in summary:
            s = summary[k]
            L.append(f"{label}: {s['trades']} trades, net ${s['net_pnl']:,.0f}, PF {s['profit_factor']:.2f}, win {s['win_rate']:.1%}")
    if "by_year" in summary:
        L += ["", "By year: " + ", ".join(f"{y}: ${v['net_pnl']:,.0f} ({v['trades']})" for y, v in summary["by_year"].items())]
    if o["trades"] < 30:
        L += ["", "WARNING: fewer than 30 trades — results are not statistically meaningful."]
    return "\n".join(L)
