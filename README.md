# GC // BACKTEST

Paste a YouTube trading-strategy video. Get that strategy backtested on **GC (gold futures)**, with win rate shown for the **5-minute, 15-minute and 1-hour** timeframes over the **last week or last month**.

```
YouTube link -> transcript -> strategy spec (JSON) -> backtest engine -> dashboard
```

Claude reads the transcript and fills in a strict rule schema. It never writes backtest code. The rules are then run by a small, tested engine built on pandas and NumPy.

> **Not financial advice.** Backtests are simulations. Past results do not predict future results, and a backtest cannot capture every real-world fill. Use this to test ideas, not to size real positions.

---

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/<you>/gc-backtester.git
cd gc-backtester
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

export ANTHROPIC_API_KEY=your-key-here                  # or skip the key and use a free local model (see below)
gcbt ui                                                 # open http://127.0.0.1:8765
```

No API key yet? Press **DEMO** in the dashboard, or run the offline demo:

```bash
gcbt run --spec gcbt/examples/ema_cross.json --source synthetic
```

The synthetic source is a random walk and exists only to prove the pipeline works. Its results mean nothing.

## Dashboard

`gcbt ui` serves a local page (bound to `127.0.0.1` only, never exposed to your network).

- Paste a YouTube link and press **RUN**. If YouTube won't hand over captions (captions off, or you're on a cloud/VPN IP), a box appears where you can paste the transcript instead (YouTube: `...` menu -> *Show transcript*).
- Three rings show win rate for **5m, 15m and 1h**. Toggle **Last week / Last month** without reloading.
- Each card also shows trade count, profit factor, net P&L and a P&L curve.
- Cards with fewer than 10 trades are marked **LOW SAMPLE**.
- Any part of the video's strategy that could not be turned into rules is listed under the cards, so you know what was *not* tested.

## Command line

```bash
# From a video (needs ANTHROPIC_API_KEY). Saves the extracted spec so you can review it.
gcbt run --url "https://youtu.be/XXXXXXXXXXX" --save-spec spec.json

# From a spec you already have, or a transcript you pasted into a file
gcbt run --spec spec.json
gcbt run --transcript transcript.txt

# Use your own data: a 5-minute TradingView export, resampled to --tf
gcbt run --spec spec.json --source csv --csv gc_5m.csv --tf 15m

# Micro gold, custom costs
gcbt run --spec spec.json --symbol MGC --commission 1.0 --slippage-ticks 2
```

Output goes to `out/`: `trades.csv`, `summary.json`, `equity.png`, plus a text report in the terminal that includes in-sample vs out-of-sample results and a per-year breakdown.

Videos with no captions: add `--whisper` (install `pip install -e ".[whisper]"` and ffmpeg), or paste the transcript into a file and use `--transcript`.

## Running it on GitHub

| What | How |
|---|---|
| **Tests** | Run automatically on every push (`.github/workflows/tests.yml`). |
| **Dashboard in your browser (Codespaces)** | Repo page -> **Code** -> **Codespaces** -> **Create codespace**. It installs everything and opens the dashboard on port 8765. Add your key first under *Settings -> Codespaces -> Secrets* as `ANTHROPIC_API_KEY`. |
| **Scheduled backtest** | `.github/workflows/scheduled-backtest.yml` backtests a spec on live GC data every Monday (or on demand from the **Actions** tab -> *Run workflow*) and posts the 5m / 15m / 1h win-rate table to the run summary. Put your own specs in `strategies/` and point the workflow at one. |

GitHub Pages can't host the dashboard because it needs the Python backend.

Cloud IPs (Codespaces, Actions) are often blocked by YouTube, so caption fetching may fail there. Paste the transcript into the dashboard's fallback box, or extract the spec locally and commit the resulting JSON to `strategies/`. Yahoo can also rate-limit cloud IPs; if the scheduled job fails for that reason, rerun it later or feed it a CSV.

The same table is available locally:

```bash
gcbt mtf --spec strategies/example.json --source yfinance
```

## No API key? Use a free local model (Ollama)

The only step that needs an AI model is reading the transcript. You can run that on your own computer for free.

1. Install Ollama from **ollama.com/download** (Windows, Mac and Linux). It runs in the background after install.
2. Download a model (one time, about 4.7 GB):
   ```
   ollama pull qwen2.5:7b
   ```
   Low on memory (under 8 GB RAM)? Use a smaller one: `ollama pull qwen2.5:3b`, then set `OLLAMA_MODEL=qwen2.5:3b`
   (Windows: `set OLLAMA_MODEL=qwen2.5:3b`; Mac/Linux: `export OLLAMA_MODEL=qwen2.5:3b`).
3. Check your setup: `gcbt doctor`
4. Run as normal (`gcbt ui`). With no `ANTHROPIC_API_KEY` set, the tool uses Ollama automatically. The dashboard shows which model wrote the rules ("rules from: ollama:..."). Force a choice with `--llm ollama|anthropic` or `GCBT_LLM`.

Expect it to be slower than Claude (a minute or several on a normal PC) and less reliable on messy videos. Small models misread strategies more often, and a spec can be *valid* and still *wrong*. The checks catch malformed output, not misunderstandings, so read the extracted rules before trusting any result. The "not codified" list matters even more here.

## Always review the extracted spec

Extraction is the weakest link. Before you trust a result, open `spec.json` and check that the rules match what the video actually says. The model is told to list anything it can't express as rules under `uncodifiable` and to lower its `confidence`, but it can still misread a video. Many YouTube strategies are discretionary ("I just feel the level") and cannot be backtested faithfully.

## The strategy spec

The model returns JSON in this shape (full schema in `gcbt/spec.py` and `gcbt/extract.py`):

```json
{
  "name": "EMA 9/21 cross",
  "timeframe": "15m",
  "direction": "both",
  "long_entry": {"all": [
    {"left": {"ind": "ema", "period": 9}, "op": "crosses_above", "right": {"ind": "ema", "period": 21}}
  ]},
  "stop":   {"kind": "atr_multiple", "value": 1.5, "atr_period": 14},
  "target": {"kind": "atr_multiple", "value": 2.0},
  "uncodifiable": [],
  "confidence": "high"
}
```

- **Indicators:** `sma`, `ema`, `rsi`, `atr`, `bbands`, `macd`, `donchian`, `vwap`, `open/high/low/close`
- **Operators:** `>`, `<`, `>=`, `<=`, `crosses_above`, `crosses_below`
- **Stops/targets:** `fixed_points`, `atr_multiple`, `percent`
- **Optional:** `session_utc` trading window, `contracts`

To add an indicator, write it in `gcbt/indicators.py`, wire it up in `gcbt/signals.py`, and add it to the allow-list in `gcbt/spec.py` and the prompt in `gcbt/extract.py`.

## How the backtest stays honest

These rules are covered by tests in `tests/test_engine.py`:

- Signals are computed on a bar's close and **filled at the next bar's open**. No lookahead.
- Stops and targets are checked intrabar. If one bar touches both, the **stop** is assumed to hit first.
- A gap through a stop fills at the **open**, not at the stop price.
- Market fills pay slippage (default 1 tick). Commission is charged per side (default $2.50).
- Contract math: **GC** = 100 oz, $10 per 0.10 tick. **MGC** = 10 oz, $1 per tick.
- Indicators are causal: signals on truncated data equal the prefix of signals on the full data.

## Data sources

| Source | Notes |
|---|---|
| `yfinance` (`GC=F`) | Fine for prototyping. Roughly 2 months of 5m/15m and 2 years of 1h (Yahoo's limits). Continuous contract, so roll gaps are **not** back-adjusted. |
| TradingView CSV | Chart menu -> *Export chart data*, on a 5-minute chart. 15m and 1h are resampled from it. Bar count depends on your plan. |
| Databento, Interactive Brokers, Polygon | Better for serious work. Add a loader in `gcbt/data.py` that returns a UTC-indexed OHLCV DataFrame. |

TradingView has no public historical-data API. Scraping libraries break often and are against its terms, so this project doesn't use them.

## Limitations

- **Periods are in bars.** "EMA 20" is 100 minutes on 5m but 20 hours on 1h, so the same rules behave differently on each timeframe. The dashboard runs the identical rules on all three.
- **Small samples.** A week of 1h bars often produces only a few trades. Treat low-sample numbers as noise.
- **Win rate is not edge.** A 35% win rate with a 2:1 payoff can be profitable. Read it alongside profit factor and net P&L.
- **Overfitting and regimes.** A strategy that works in one gold regime can fail in another. Check the per-year breakdown and the out-of-sample split.
- **Roll handling.** Use properly back-adjusted data for long histories.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

Tests run on every push via GitHub Actions (`.github/workflows/tests.yml`).

```
gcbt/
  transcript.py   YouTube -> text (captions, optional Whisper)
  extract.py      text -> validated spec (Claude API)
  spec.py         schema and validation
  indicators.py   SMA, EMA, RSI, ATR, Bollinger, MACD, Donchian, VWAP
  signals.py      spec rules -> boolean signals
  engine.py       event-driven backtester (GC / MGC)
  metrics.py      P&L, win rate, profit factor, Sharpe, drawdown, IS/OOS, by-year
  mtf.py          5m / 15m / 1h runs and week / month windows
  server.py       local dashboard API
  ui/index.html   dashboard
strategies/       your saved specs (used by the scheduled GitHub job)
.devcontainer/    Codespaces setup
```

## License

MIT. See [LICENSE](LICENSE).
