"""Local web server for the dashboard. Stdlib only. Binds to 127.0.0.1 — it is not meant to be exposed."""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .extract import backend_label, extract_spec
from .mtf import run_multi
from .spec import SpecError, StrategySpec
from .transcript import get_transcript

UI = Path(__file__).parent / "ui" / "index.html"
DEMO = Path(__file__).parent / "examples" / "ema_cross.json"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, default=str).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/run":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            source = req.get("source", "synthetic")
            by = "provided spec"
            if req.get("demo"):
                spec = StrategySpec.from_dict(json.loads(DEMO.read_text()))
                by = "built-in demo"
            elif req.get("spec"):
                spec = StrategySpec.from_dict(req["spec"])
            elif req.get("transcript"):  # pasted text: the fallback when captions can't be fetched
                by = backend_label()
                spec, _ = extract_spec(str(req["transcript"]))
            elif req.get("url"):
                by = backend_label()  # fail fast if no language model is available
                spec, _ = extract_spec(get_transcript(req["url"], allow_whisper=False))
            else:
                return self._json(400, {"error": "Provide a YouTube url, a transcript, a spec, or demo:true."})
            out = run_multi(spec, source, req.get("csv"))
            out["strategy"]["extracted_by"] = by
            self._json(200, out)
        except (SpecError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as e:
            self._json(400, {"error": f"{type(e).__name__}: {e}"})
        except Exception as e:  # keep the UI alive on unexpected failures
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        pass


def serve(port: int = 8765, host: str = "127.0.0.1") -> None:
    """Default is loopback only. Use host=0.0.0.0 only inside a container that forwards the port privately (Codespaces)."""
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"GC//BACKTEST dashboard -> http://{host}:{port}", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
