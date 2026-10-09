"""Ollama backend, tested against a stand-in HTTP server that speaks Ollama's API."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from gcbt import cli, extract
from gcbt.spec import SpecError

DEMO = json.loads((Path(__file__).parent.parent / "gcbt" / "examples" / "ema_cross.json").read_text())


@pytest.fixture()
def ollama(monkeypatch):
    state = {"models": ["qwen2.5:7b"], "replies": [], "requests": []}

    class H(BaseHTTPRequestHandler):
        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"models": [{"name": m} for m in state["models"]]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(req)
            self._send({"message": {"role": "assistant", "content": state["replies"].pop(0)}})

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{srv.server_address[1]}")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GCBT_LLM", raising=False)
    yield state
    srv.shutdown()


def test_auto_picks_ollama_without_key(ollama):
    assert extract.choose_backend() == "ollama"
    assert extract.backend_label().startswith("ollama:")


def test_auto_prefers_claude_when_key_set(ollama, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert extract.choose_backend() == "anthropic"
    assert extract.choose_backend("ollama") == "ollama"


def test_extract_via_ollama_sends_json_mode_and_system_prompt(ollama):
    ollama["replies"].append(json.dumps(DEMO))
    spec, _ = extract.extract_spec("buy when 9 ema crosses 21 ema")
    assert spec.name == DEMO["name"]
    req = ollama["requests"][0]
    assert req["format"] == "json" and req["options"]["temperature"] == 0 and req["stream"] is False
    assert req["messages"][0]["role"] == "system" and "STRICT JSON" in req["messages"][0]["content"]


def test_small_model_gets_two_repair_attempts(ollama):
    bad = dict(DEMO, long_entry={"all": [{"left": {"ind": "supply_zone"}, "op": ">", "right": 1}]})
    ollama["replies"] += ["I think the strategy is...", json.dumps(bad), json.dumps(DEMO)]
    spec, _ = extract.extract_spec("transcript")
    assert len(ollama["requests"]) == 3 and spec.confidence == "high"


def test_garbage_json_types_do_not_crash(ollama):
    ollama["replies"] += ['["not", "an", "object"]', "[1]", "[2]"]  # regex finds {} in none -> SpecError path
    with pytest.raises(SpecError):
        extract.extract_spec("transcript")


def test_missing_model_gives_pull_command(ollama):
    ollama["models"] = ["llama3.1:8b"]
    with pytest.raises(RuntimeError, match="ollama pull qwen2.5:7b"):
        extract.extract_spec("transcript")


def test_no_backend_available_explains_both_options(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GCBT_LLM", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:1")  # nothing listening
    with pytest.raises(RuntimeError) as e:
        extract.extract_spec("transcript")
    assert "ANTHROPIC_API_KEY" in str(e.value) and "ollama pull" in str(e.value)


def test_doctor_reports_ollama(ollama, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_reachable", lambda host: True)
    cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "Ollama running" in out and "installed" in out and "A language model is available" in out


def test_dashboard_end_to_end_with_local_model(ollama):
    """Pasted transcript -> Ollama -> spec -> 5m/15m/1h backtest, no API key and no internet."""
    from urllib import request as urlrequest
    from gcbt import server

    ollama["replies"].append(json.dumps(DEMO))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        req = urlrequest.Request(f"http://127.0.0.1:{srv.server_address[1]}/api/run",
                                 json.dumps({"transcript": "ema 9 crosses ema 21", "source": "synthetic"}).encode(),
                                 {"Content-Type": "application/json"})
        out = json.loads(urlrequest.urlopen(req).read())
    finally:
        srv.shutdown()
    assert out["strategy"]["extracted_by"] == "ollama:qwen2.5:7b"
    assert set(out["results"]) == {"5m", "15m", "1h"}
