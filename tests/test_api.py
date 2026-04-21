"""Smoke-test the FastAPI surface via TestClient. Skipped if deps missing."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("librosa")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import soundfile as sf  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from songfuse.api import app  # noqa: E402

SR = 22_050


def _make_tone(path, freq: float, bpm: int = 120, seconds: float = 4.0) -> None:
    n = int(SR * seconds)
    t = np.arange(n) / SR
    y = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    step = int(SR * 60 / bpm)
    for s in range(0, n, step):
        y[s : s + int(SR * 0.02)] += 0.6
    sf.write(str(path), y, SR, subtype="PCM_16")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def tracks(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _make_tone(a, 220.0, bpm=120)
    _make_tone(b, 330.0, bpm=124)
    return a, b


def test_index_serves_web_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Songfuse" in r.text


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_analyze_endpoint(client, tracks):
    a, _ = tracks
    with open(a, "rb") as f:
        r = client.post("/analyze", files={"file": ("a.wav", f, "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    for key in ("bpm", "key", "camelot", "duration", "segments"):
        assert key in body


def test_fuse_returns_wav_with_plan_header(client, tracks):
    a, b = tracks
    with open(a, "rb") as fa, open(b, "rb") as fb:
        r = client.post(
            "/fuse",
            files={
                "track_a": ("a.wav", fa, "audio/wav"),
                "track_b": ("b.wav", fb, "audio/wav"),
            },
            data={"style": "equal_power", "bars": "2", "lufs": "-20", "match_key": "false"},
        )
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
    assert r.content[8:12] == b"WAVE"
    plan = json.loads(r.headers["x-songfuse-plan"])
    assert plan["plan"]["target_bpm"] > 0


def test_fuse_rejects_bad_style(client, tracks):
    a, b = tracks
    with open(a, "rb") as fa, open(b, "rb") as fb:
        r = client.post(
            "/fuse",
            files={
                "track_a": ("a.wav", fa, "audio/wav"),
                "track_b": ("b.wav", fb, "audio/wav"),
            },
            data={"style": "definitely_not_a_style", "bars": "2"},
        )
    assert r.status_code == 400
