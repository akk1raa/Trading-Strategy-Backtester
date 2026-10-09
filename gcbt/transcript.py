"""YouTube link -> transcript text. Captions first; yt-dlp + Whisper as an optional fallback."""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})")


def video_id(url: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    m = _ID.search(url)
    if not m:
        raise ValueError(f"Could not find a YouTube video id in: {url}")
    return m.group(1)


def fetch_captions(url: str, languages: tuple[str, ...] = ("en", "en-US", "en-GB")) -> str:
    from youtube_transcript_api import YouTubeTranscriptApi

    vid = video_id(url)
    api = YouTubeTranscriptApi()  # v1.x API; older versions used the static get_transcript
    if hasattr(api, "fetch"):
        parts = api.fetch(vid, languages=list(languages))
        return " ".join(s.text for s in parts)
    parts = YouTubeTranscriptApi.get_transcript(vid, languages=list(languages))  # pragma: no cover
    return " ".join(p["text"] for p in parts)


def transcribe_with_whisper(url: str, model: str = "base") -> str:
    """Fallback when a video has no captions. Needs `pip install yt-dlp openai-whisper` and ffmpeg."""
    import whisper  # lazy: heavy optional dependency

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "audio.%(ext)s"
        subprocess.run(["yt-dlp", "-x", "--audio-format", "mp3", "-o", str(out), url], check=True)
        audio = next(Path(td).glob("audio.*"))
        return whisper.load_model(model).transcribe(str(audio))["text"]


def get_transcript(url: str, allow_whisper: bool = False) -> str:
    try:
        return fetch_captions(url)
    except Exception as e:  # captions disabled, blocked IP, no English track...
        if not allow_whisper:
            raise RuntimeError(
                f"Could not fetch captions ({type(e).__name__}). Videos with captions off, and many cloud/VPN "
                "IPs, are blocked by YouTube. Paste the transcript instead (dashboard: the box that appears; "
                "CLI: --transcript file.txt), or on the CLI try --whisper."
            ) from e
        return transcribe_with_whisper(url)
