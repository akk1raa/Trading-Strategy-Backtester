"""Tests for the parts that normally need the network or an API key, using stand-ins."""
import json
import threading
import types
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest

from gcbt import data, extract, server, transcript
from gcbt.spec import SpecError

DEMO = json.loads((Path(__file__).parent.parent / "gcbt" / "examples" / "ema_cross.json").read_text())


# ---------- YouTube link parsing ----------
@pytest.mark.parametrize("url", [
    "https://youtu.be/RnP08K2SAZs?si=La-L3h7RR2z5oph1",
    "https://www.youtube.com/watch?v=RnP08K2SAZs&t=42s",
    "https://www.youtube.com/shorts/RnP08K2SAZs",
    "https://www.youtube.com/live/RnP08K2SAZs",
    "RnP08K2SAZs",
])
def test_video_id_formats(url):
    assert transcript.video_id(url) == "RnP08K2SAZs"


def test_video_id_rejects_garbage():
    with pytest.raises(ValueError):
        transcript.video_id("https://example.com/not-youtube")


def test_caption_failure_gives_actionable_error(monkeypatch):
    def boom(url):
        raise ConnectionError("blocked")
    monkeypatch.setattr(transcript, "fetch_captions", boom)
    with pytest.raises(RuntimeError, match="Paste the transcript"):
        transcript.get_transcript("RnP08K2SAZs")


# ---------- LLM extraction (fake client) ----------
class FakeClient:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls += 1
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=self.replies.pop(0))])


def test_extract_accepts_json_wrapped_in_prose():
    c = FakeClient("Here you go:\n```json\n" + json.dumps(DEMO) + "\n```")
    spec, raw = extract.extract_spec("transcript", client=c)
    assert spec.name == DEMO["name"] and c.calls == 1


def test_extract_repairs_one_bad_reply():
    bad = dict(DEMO, long_entry={"all": [{"left": {"ind": "supply_zone"}, "op": ">", "right": 1}]})
    c = FakeClient(json.dumps(bad), json.dumps(DEMO))
    spec, _ = extract.extract_spec("transcript", client=c)
    assert c.calls == 2 and spec.confidence == "high"


def test_extract_gives_up_after_two_bad_replies():
    c = FakeClient("not json at all", "still not json")
    with pytest.raises(SpecError):
        extract.extract_spec("transcript", client=c)


def test_extract_needs_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GCBT_LLM", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:1")  # make sure no local Ollama is picked up
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        extract.extract_spec("transcript")


# ---------- data loaders ----------
def _ohlcv(n=50, freq="5min"):
    idx = pd.date_range("2024-03-01", periods=n, freq=freq, tz="UTC")
    c = 2000 + np.cumsum(np.random.default_rng(1).normal(0, .3, n))
    return pd.DataFrame({"open": c, "high": c + .5, "low": c - .5, "close": c, "volume": 100.0}, index=idx)


def test_tradingview_csv_unix_and_iso_times(tmp_path):
    df = _ohlcv()
    secs = (df.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)  # unit-proof
    cols = {k: df[k].values for k in df}
    pd.DataFrame({"time": secs, **cols}).to_csv(tmp_path / "s.csv", index=False)
    pd.DataFrame({"time": secs * 1000, **cols}).to_csv(tmp_path / "ms.csv", index=False)
    pd.DataFrame({"time": secs * 10**9, **cols}).to_csv(tmp_path / "ns.csv", index=False)
    pd.DataFrame({"time": df.index.strftime("%Y-%m-%dT%H:%M:%SZ"), "open": df.open.values, "high": df.high.values,
                  "low": df.low.values, "close": df.close.values, "Volume": df.volume.values}).to_csv(tmp_path / "i.csv", index=False)
    for f in ("s.csv", "ms.csv", "ns.csv", "i.csv"):
        out = data.load_tradingview_csv(str(tmp_path / f))
        assert len(out) == 50 and str(out.index.tz) == "UTC", f
        assert out.index[0] == df.index[0] and out.close.iloc[0] == pytest.approx(df.close.iloc[0]), f


def test_resample_aggregates_correctly():
    df = _ohlcv(60)
    h = data.resample(df, "1h")
    first = df.iloc[:12]
    assert h.iloc[0].open == first.open.iloc[0] and h.iloc[0].high == first.high.max()
    assert h.iloc[0].low == first.low.min() and h.iloc[0].close == first.close.iloc[-1]


def test_yfinance_request_shape_and_cleanup(monkeypatch):
    """Real yfinance returns MultiIndex columns and a New York tz index; 'period' must not be a made-up '728d'."""
    seen = {}
    base = _ohlcv(30, "1h").tz_convert("America/New_York")
    base.columns = pd.MultiIndex.from_product([[c.capitalize() for c in base.columns], ["GC=F"]], names=["Price", "Ticker"])

    def fake_download(symbol, **kw):
        seen.update(kw)
        return base

    import yfinance
    monkeypatch.setattr(yfinance, "download", fake_download)
    df = data.load_yfinance("GC=F", "1h")
    assert "period" not in seen and "start" in seen and seen["interval"] == "1h"
    assert list(df.columns) == data.OHLCV and str(df.index.tz) == "UTC"


# ---------- dashboard API ----------
@pytest.fixture()
def api():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(body):
        req = urlrequest.Request(base + "/api/run", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urlrequest.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except HTTPError as e:
            return e.code, json.loads(e.read())

    yield base, post
    srv.shutdown()


def test_api_demo_returns_all_timeframes_and_windows(api):
    _, post = api
    code, out = post({"demo": True, "source": "synthetic"})
    assert code == 200
    assert set(out["results"]) == {"5m", "15m", "1h"}
    assert all({"week", "month"} <= set(v) for v in out["results"].values())


def test_api_url_with_blocked_captions_returns_clear_error(api, monkeypatch):
    _, post = api
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(server, "get_transcript", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Could not fetch captions (X)")))
    code, out = post({"url": "RnP08K2SAZs"})
    assert code == 400 and "captions" in out["error"]


def test_api_pasted_transcript_path(api, monkeypatch):
    _, post = api
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(server, "extract_spec", lambda text: (server.StrategySpec.from_dict(DEMO), DEMO))
    code, out = post({"transcript": "buy when the 9 ema crosses the 21 ema", "source": "synthetic"})
    assert code == 200 and out["strategy"]["name"] == DEMO["name"]


def test_api_rejects_bad_spec_and_empty_request(api):
    _, post = api
    assert post({})[0] == 400
    code, out = post({"spec": dict(DEMO, direction="sideways"), "source": "synthetic"})
    assert code == 400 and "direction" in out["error"]


def test_dashboard_html_served(api):
    base, _ = api
    with urlrequest.urlopen(base + "/") as r:
        html = r.read().decode()
    assert "GC" in html and 'id="runT"' in html


# ---------- mtf command / GitHub job ----------
def test_mtf_cli_prints_table_and_writes_json(tmp_path, capsys):
    from gcbt import cli
    spec_path = Path(__file__).parent.parent / "strategies" / "example.json"
    cli.main(["mtf", "--spec", str(spec_path), "--source", "synthetic", "--json-out", str(tmp_path / "m.json")])
    out = capsys.readouterr().out
    assert "| 5m | week |" in out and "| 1h | month |" in out and "Not financial advice" in out
    assert set(json.loads((tmp_path / "m.json").read_text())["results"]) == {"5m", "15m", "1h"}


def test_mtf_cli_bad_spec_exits_cleanly(tmp_path):
    from gcbt import cli
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(dict(DEMO, direction="sideways")))
    with pytest.raises(SystemExit) as e:
        cli.main(["mtf", "--spec", str(p), "--source", "synthetic"])
    assert e.value.code == 1


def test_repo_config_files_are_valid():
    root = Path(__file__).parent.parent
    dc = json.loads((root / ".devcontainer" / "devcontainer.json").read_text())
    assert 8765 in dc["forwardPorts"] and "0.0.0.0" in dc["postAttachCommand"]
    assert json.loads((root / "strategies" / "example.json").read_text())["name"]
    yaml = pytest.importorskip("yaml")
    for wf in (root / ".github" / "workflows").glob("*.yml"):
        doc = yaml.safe_load(wf.read_text())
        assert doc["jobs"], wf.name
