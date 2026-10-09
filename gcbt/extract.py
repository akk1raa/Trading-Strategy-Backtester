"""Transcript -> validated StrategySpec, via the Claude API. The LLM fills a schema; it never writes backtest code."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from .spec import SpecError, StrategySpec

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM = """You convert trading-strategy video transcripts into a STRICT JSON spec for a backtester.
Output ONLY one JSON object, no prose, no code fences.

Schema:
{
  "name": str,
  "timeframe": "5m"|"15m"|"1h"|"1d",            // the timeframe the video trades; best guess if unstated
  "direction": "long"|"short"|"both",
  "long_entry":  {"all": [Condition...], "any": [Condition...]},
  "long_exit":   {"all": [...], "any": [...]},    // may be empty; stops/targets can do the exiting
  "short_entry": {"all": [...], "any": [...]},
  "short_exit":  {"all": [...], "any": [...]},
  "stop":   {"kind": "none"|"fixed_points"|"atr_multiple"|"percent", "value": number, "atr_period": 14},
  "target": {"kind": "none"|"fixed_points"|"atr_multiple"|"percent", "value": number, "atr_period": 14},
  "session_utc": [startHour, endHour] | null,     // only if the video restricts trading hours (convert to UTC)
  "contracts": 1,
  "uncodifiable": [str],                          // EVERYTHING you could not express as rules (see below)
  "confidence": "high"|"medium"|"low"
}
Condition = {"left": Operand, "op": ">"|"<"|">="|"<="|"crosses_above"|"crosses_below", "right": Operand}
Operand = number | {"ind": NAME, ...params}
NAME in: sma, ema (period, source), rsi (period), atr (period),
         bbands (period, std, field: upper|mid|lower), macd (fast, slow, signal, field: line|signal|hist),
         donchian (period, field: upper|lower), vwap, close, open, high, low

Rules:
- Use ONLY the indicators/operators above. If the video uses something else (order blocks, supply/demand zones,
  "I feel the level", news, volume profile, fibonacci, candlestick patterns), do NOT approximate it silently:
  add a clear line to "uncodifiable" and lower "confidence".
- If a stop/target is described only vaguely ("tight stop"), put it in "uncodifiable" and use kind "none".
- Never invent parameters the speaker did not state, except standard defaults (e.g. RSI 14); mention defaults in "uncodifiable".
- If the strategy is entirely discretionary, return empty entry rules and explain in "uncodifiable"."""


def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise SpecError("Model did not return JSON.")
    return json.loads(m.group(0))


# ---------------------------------------------------------------- backends
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")


def ollama_url() -> str:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    return host if host.startswith("http") else f"http://{host}"


def _ollama_get(path: str, timeout: float = 3.0) -> dict:
    with urllib.request.urlopen(ollama_url() + path, timeout=timeout) as r:
        return json.loads(r.read())


def ollama_models() -> list[str] | None:
    """Names of locally installed models, or None if the Ollama server is not reachable."""
    try:
        return [m["name"] for m in _ollama_get("/api/tags").get("models", [])]
    except (OSError, ValueError):
        return None


def _model_installed(wanted: str, installed: list[str]) -> bool:
    return wanted in installed or (":" not in wanted and f"{wanted}:latest" in installed)


def choose_backend(preferred: str | None = None) -> str:
    """'anthropic' or 'ollama'. 'auto' prefers Claude when a key is set, then a running local Ollama."""
    pref = (preferred or os.environ.get("GCBT_LLM") or "auto").lower()
    if pref not in ("auto", "anthropic", "ollama"):
        raise ValueError(f"Unknown LLM backend '{pref}'. Use auto, anthropic or ollama.")
    if pref == "anthropic" or (pref == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY, or use a local model (see the README: Ollama).")
        return "anthropic"
    models = ollama_models()
    if models is None:
        raise RuntimeError(
            "No language model available to read the transcript. Either set ANTHROPIC_API_KEY, or install Ollama "
            f"(ollama.com), run `ollama pull {OLLAMA_MODEL}`, keep Ollama running, and try again. "
            "Run `gcbt doctor` to check your setup."
        )
    return "ollama"


def backend_label(preferred: str | None = None) -> str:
    b = choose_backend(preferred)
    return f"ollama:{OLLAMA_MODEL}" if b == "ollama" else f"claude:{MODEL}"


def _ollama_complete(system: str, messages: list[dict], timeout: float = 900.0) -> str:
    installed = ollama_models()
    if installed is None:
        raise RuntimeError("Ollama is not running. Start the Ollama app, then try again.")
    if not _model_installed(OLLAMA_MODEL, installed):
        raise RuntimeError(f"Ollama model '{OLLAMA_MODEL}' is not installed. Run: ollama pull {OLLAMA_MODEL}")
    body = json.dumps({
        "model": OLLAMA_MODEL, "stream": False, "format": "json",
        "messages": [{"role": "system", "content": system}, *messages],
        "options": {"temperature": 0, "num_ctx": 16384},
    }).encode()
    req = urllib.request.Request(ollama_url() + "/api/chat", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["message"]["content"]
    except (OSError, KeyError, ValueError) as e:  # URLError, timeouts, bad payloads
        raise RuntimeError(f"Ollama request failed ({type(e).__name__}: {e}). Local models can be slow; "
                           "try a smaller OLLAMA_MODEL such as qwen2.5:3b.") from e


def _claude_complete(client, system: str, messages: list[dict]) -> str:
    resp = client.messages.create(model=MODEL, max_tokens=4000, system=system, messages=messages)
    return resp.content[0].text


def extract_spec(transcript: str, max_chars: int = 60_000, client=None, backend: str | None = None) -> tuple[StrategySpec, dict]:
    """Transcript -> validated spec. `client` (an Anthropic-style client) is mainly for tests."""
    if client is not None:
        complete, attempts = (lambda sys_, msgs: _claude_complete(client, sys_, msgs)), 2
    elif choose_backend(backend) == "ollama":
        # Local models get a shorter transcript (context is limited) and one extra repair attempt.
        max_chars, attempts = min(max_chars, 30_000), 3
        complete = _ollama_complete
    else:
        import anthropic

        c = anthropic.Anthropic()
        complete, attempts = (lambda sys_, msgs: _claude_complete(c, sys_, msgs)), 2

    messages = [{"role": "user", "content": f"Transcript:\n\n{transcript[:max_chars]}"}]
    last_err = None
    for _ in range(attempts):
        text = complete(SYSTEM, messages)
        try:
            raw = _parse_json(text)
            return StrategySpec.from_dict(raw), raw
        except (SpecError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            last_err = e
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"Invalid: {e}. Return corrected JSON only, following the schema exactly."}]
    raise SpecError(f"Could not get a valid spec from the model: {last_err}. "
                    "Try again, use a stronger model, or write the spec by hand (see strategies/example.json).")
