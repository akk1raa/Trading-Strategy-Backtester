"""CLI:  python -m gcbt ui | run | extract"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import load_tradingview_csv, load_yfinance, norm_tf, resample, synthetic
from .engine import CostConfig, run_backtest
from .extract import backend_label, extract_spec, ollama_models, OLLAMA_MODEL
from .metrics import format_report, summarize
from .spec import SpecError, StrategySpec
from .transcript import get_transcript


def _spec_from_args(a) -> StrategySpec:
    if a.spec:
        return StrategySpec.from_dict(json.loads(Path(a.spec).read_text()))
    text = Path(a.transcript).read_text() if a.transcript else get_transcript(a.url, allow_whisper=a.whisper)
    spec, raw = extract_spec(text, backend=getattr(a, 'llm', None))
    print(f'Extracted with {backend_label(getattr(a, "llm", None))}', file=sys.stderr)
    Path(a.save_spec or "spec.json").write_text(json.dumps(raw, indent=2))
    print(f"Saved extracted spec -> {a.save_spec or 'spec.json'} (review it before trusting results)", file=sys.stderr)
    return spec


def _load(a, tf: str):
    if a.source == "csv":
        return resample(load_tradingview_csv(a.csv), tf)
    if a.source == "synthetic":
        return synthetic(tf=tf)
    return load_yfinance("GC=F", tf, a.start, a.end)


def _reachable(host: str) -> bool:
    """A real HTTPS request (honours proxies). Any HTTP reply from the site counts; tunnel/DNS/TLS failures do not."""
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(urllib.request.Request(f"https://{host}/", method="HEAD"), timeout=8).close()
        return True
    except urllib.error.HTTPError:
        return True  # the site answered (even with 404/405), so the network path works
    except Exception:
        return False


def _doctor() -> None:
    import os

    ok = lambda b: "OK  " if b else "FAIL"  # noqa: E731
    print(f"{ok(sys.version_info >= (3, 10))} Python {sys.version.split()[0]} (needs 3.10+)")
    key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    print(f"{'OK  ' if key else 'none'} ANTHROPIC_API_KEY {'is set' if key else 'not set (fine if you use Ollama)'}")
    models = ollama_models()
    if models is None:
        print("none Ollama not running (install from ollama.com if you want a free local model)")
    else:
        have = OLLAMA_MODEL in models or f"{OLLAMA_MODEL}:latest" in models
        print(f"{ok(have)} Ollama running; wanted model '{OLLAMA_MODEL}' {'installed' if have else 'NOT installed -> ollama pull ' + OLLAMA_MODEL}")
    print(f"{ok(key or (models is not None))} A language model is available to read transcripts")
    for label, host in (("YouTube", "www.youtube.com"), ("Yahoo Finance (GC data)", "query1.finance.yahoo.com")):
        print(f"{ok(_reachable(host))} Can reach {label}")


def main(argv=None) -> None:
    try:
        _main(argv)
    except (SpecError, RuntimeError, ValueError, FileNotFoundError) as e:  # clean message, no traceback
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


def _main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="gcbt")
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("ui", help="launch the dashboard")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 only inside a container (e.g. Codespaces)")

    sub.add_parser("doctor", help="check your setup (Python, API key, Ollama, internet)")

    m = sub.add_parser("mtf", help="win rate on 5m/15m/1h for the last week and month (markdown table)")
    m.add_argument("--spec", required=True)
    m.add_argument("--source", choices=["yfinance", "csv", "synthetic"], default="yfinance")
    m.add_argument("--csv"); m.add_argument("--symbol", choices=["GC", "MGC"], default="GC")
    m.add_argument("--json-out", help="also write the raw results as JSON")

    for name in ("run", "extract"):
        r = sub.add_parser(name)
        g = r.add_mutually_exclusive_group(required=True)
        g.add_argument("--url"); g.add_argument("--transcript"); g.add_argument("--spec")
        r.add_argument("--whisper", action="store_true", help="transcribe audio if no captions")
        r.add_argument("--llm", choices=["auto", "anthropic", "ollama"], default=None, help="which model reads the transcript")
        r.add_argument("--save-spec")
    r = sub.choices["run"]
    r.add_argument("--source", choices=["yfinance", "csv", "synthetic"], default="yfinance")
    r.add_argument("--csv"); r.add_argument("--start"); r.add_argument("--end")
    r.add_argument("--tf", help="override the spec's timeframe")
    r.add_argument("--symbol", choices=["GC", "MGC"], default="GC")
    r.add_argument("--commission", type=float, default=2.5)
    r.add_argument("--slippage-ticks", type=float, default=1.0)
    r.add_argument("--out", default="out")

    a = p.parse_args(argv)
    if a.cmd == "ui":
        from .server import serve
        return serve(a.port, a.host)
    if a.cmd == "doctor":
        return _doctor()
    if a.cmd == "mtf":
        from .mtf import format_markdown, run_multi
        spec = StrategySpec.from_dict(json.loads(Path(a.spec).read_text()))
        res = run_multi(spec, a.source, a.csv, CostConfig(a.symbol))
        if a.json_out:
            Path(a.json_out).write_text(json.dumps(res, indent=2, default=str))
        print(format_markdown(res))
        return

    spec = _spec_from_args(a)
    if a.cmd == "extract":
        print(json.dumps(json.loads(Path(a.save_spec or "spec.json").read_text()), indent=2)); return

    tf = norm_tf(a.tf or spec.timeframe)
    df = _load(a, tf)
    cfg = CostConfig(a.symbol, a.commission, a.slippage_ticks)
    res = run_backtest(df, spec, cfg)
    summ = summarize(res)
    out = Path(a.out); out.mkdir(exist_ok=True)
    res.trades.to_csv(out / "trades.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = res.equity.plot(figsize=(10, 4), title=spec.name); ax.set_ylabel("Equity ($)")
        plt.tight_layout(); plt.savefig(out / "equity.png", dpi=140); plt.close()
    except Exception as e:  # plotting is optional
        print(f"(skipped equity plot: {e})", file=sys.stderr)
    print(format_report(spec, summ))
