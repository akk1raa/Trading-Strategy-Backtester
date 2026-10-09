import numpy as np
import pandas as pd
import pytest

from gcbt import indicators as ta
from gcbt.data import synthetic
from gcbt.engine import CostConfig, run_backtest
from gcbt.signals import build_signals
from gcbt.spec import SpecError, StrategySpec

NO_COST = CostConfig(commission_per_side=0.0, slippage_ticks=0.0)


def bars(rows):
    """rows: (open, high, low, close). Hourly UTC index."""
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1000.0)


def spec(**over):
    d = {
        "name": "t", "timeframe": "1h", "direction": "long",
        "long_entry": {"all": [{"left": {"ind": "close"}, "op": ">", "right": 100}]},
        "long_exit": {"all": [{"left": {"ind": "close"}, "op": "<", "right": 100}]},
    }
    d.update(over)
    return StrategySpec.from_dict(d)


def test_fills_on_next_bar_open_not_signal_bar():
    # bar1 close=101 triggers entry; must fill at bar2 open (102), not 101.
    df = bars([(99, 99.5, 98.5, 99), (99, 101.5, 99, 101), (102, 103, 101.5, 102.5), (102.5, 103, 99, 99.5), (99, 99.5, 98, 98.5)])
    r = run_backtest(df, spec(), NO_COST)
    t = r.trades.iloc[0]
    assert t.entry == 102 and t.entry_time == df.index[2]
    # exit signal at bar3 close (99.5<100) -> fills bar4 open = 99
    assert t.exit == 99 and t.exit_time == df.index[4]
    assert t.pnl == pytest.approx((99 - 102) * 100)


def test_stop_gap_fills_at_open_not_stop_price():
    s = spec(stop={"kind": "fixed_points", "value": 2.0})
    # enter bar2 open 102 -> stop 100. bar3 gaps open at 97 -> must fill 97, not 100.
    df = bars([(99, 99.5, 98.5, 99), (99, 101.5, 99, 101), (102, 103, 101.5, 102.5), (97, 98, 96, 97.5), (97, 98, 96, 97)])
    t = run_backtest(df, s, NO_COST).trades.iloc[0]
    assert t.reason == "stop" and t.exit == 97


def test_stop_beats_target_when_bar_touches_both():
    s = spec(stop={"kind": "fixed_points", "value": 1.0}, target={"kind": "fixed_points", "value": 1.0})
    df = bars([(99, 99.5, 98.5, 99), (99, 101.5, 99, 101), (102, 110, 90, 102), (102, 103, 101, 102)])
    t = run_backtest(df, s, NO_COST).trades.iloc[0]
    assert t.reason == "stop"


def test_costs_are_applied():
    df = bars([(99, 99.5, 98.5, 99), (99, 101.5, 99, 101), (102, 103, 101.5, 102.5), (102.5, 103, 99, 99.5), (99, 99.5, 98, 98.5)])
    free = run_backtest(df, spec(), NO_COST).trades.pnl.iloc[0]
    paid = run_backtest(df, spec(), CostConfig(commission_per_side=2.5, slippage_ticks=1)).trades.pnl.iloc[0]
    # 2 sides * (1 tick = 0.10pt * $100 = $10) + 2 sides * $2.50
    assert free - paid == pytest.approx(2 * 10 + 2 * 2.5)


def test_micro_contract_scales_pnl_by_ten():
    df = bars([(99, 99.5, 98.5, 99), (99, 101.5, 99, 101), (102, 103, 101.5, 102.5), (102.5, 103, 99, 99.5), (99, 99.5, 98, 98.5)])
    gc = run_backtest(df, spec(), CostConfig("GC", 0, 0)).trades.pnl.iloc[0]
    mgc = run_backtest(df, spec(), CostConfig("MGC", 0, 0)).trades.pnl.iloc[0]
    assert gc == pytest.approx(10 * mgc)


def test_signals_have_no_lookahead():
    """Signals on a truncated series must equal the prefix of signals on the full series."""
    df = synthetic(n=1500, tf="1h")
    sp = StrategySpec.from_dict({
        "name": "x", "timeframe": "1h", "direction": "both",
        "long_entry": {"all": [{"left": {"ind": "ema", "period": 9}, "op": "crosses_above", "right": {"ind": "ema", "period": 21}}]},
        "short_entry": {"all": [{"left": {"ind": "rsi", "period": 14}, "op": "<", "right": 30},
                                {"left": {"ind": "close"}, "op": "<", "right": {"ind": "bbands", "period": 20, "field": "lower"}}]},
    })
    full, cut = build_signals(df, sp), build_signals(df.iloc[:1000], sp)
    for k in ("long_entry", "short_entry"):
        assert (full[k][:1000] == cut[k]).all(), k


def test_indicator_sanity():
    s = pd.Series(np.arange(1.0, 101.0))
    assert ta.sma(s, 5).iloc[-1] == pytest.approx(98.0)
    r = ta.rsi(synthetic(n=500).close, 14).dropna()
    assert r.between(0, 100).all()
    assert ta.rsi(s, 14).iloc[-1] == 100.0  # monotonic up


def test_spec_rejects_bad_input():
    with pytest.raises(SpecError):
        StrategySpec.from_dict({"direction": "long", "long_entry": {}})  # no entry rules
    with pytest.raises(SpecError):
        spec(long_entry={"all": [{"left": {"ind": "magic_zone"}, "op": ">", "right": 1}]})
    with pytest.raises(SpecError):
        spec(long_entry={"all": [{"left": {"ind": "close"}, "op": "~", "right": 1}]})


def test_end_to_end_multi_timeframe_demo():
    import json
    from pathlib import Path
    from gcbt.mtf import run_multi

    d = json.loads((Path(__file__).parent.parent / "gcbt" / "examples" / "ema_cross.json").read_text())
    out = run_multi(StrategySpec.from_dict(d), source="synthetic")
    for tf in ("5m", "15m", "1h"):
        for w in ("week", "month"):
            r = out["results"][tf][w]
            assert r["trades"] >= 0 and (r["win_rate"] is None or 0 <= r["win_rate"] <= 1)
