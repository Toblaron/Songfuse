"""End-to-end integration tests using synthetic tones. Skipped if librosa missing."""

from __future__ import annotations

import numpy as np
import pytest

librosa = pytest.importorskip("librosa")

from songfuse import fuse  # noqa: E402
from songfuse.audio_io import save  # noqa: E402


def _make_tone(path, freq: float, sr: int = 22_050, seconds: float = 6.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    # Fake percussive pulse every ~0.5s so librosa finds beats.
    click = np.zeros_like(t)
    step = int(sr * 0.5)
    for i in range(0, click.size, step):
        click[i: i + int(sr * 0.02)] += 0.8
    signal = 0.5 * np.sin(2 * np.pi * freq * t) + click
    save(path, signal, sr)


def test_fuse_synthetic_tracks(tmp_path):
    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    _make_tone(a_path, 220.0)
    _make_tone(b_path, 330.0)

    out_path = tmp_path / "fused.wav"
    result = fuse(a_path, b_path, out=out_path, style="equal_power", bars=2,
                  match_key=False, sr=22_050, lufs_target=-20.0)

    assert out_path.exists()
    assert result.audio.ndim == 2
    assert result.audio.shape[-1] > 0
    assert np.isfinite(result.audio).all()
    assert float(np.max(np.abs(result.audio))) <= 1.0
    assert result.plan.target_bpm > 0


def test_fuse_creates_missing_output_dir(tmp_path):
    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    _make_tone(a_path, 220.0, seconds=4.0)
    _make_tone(b_path, 330.0, seconds=4.0)
    out_path = tmp_path / "does_not_exist_yet" / "nested" / "fused.wav"
    fuse(a_path, b_path, out=out_path, style="equal_power", bars=1,
         match_key=False, sr=22_050, lufs_target=-20.0)
    assert out_path.exists()


def test_fuse_rejects_invalid_style(tmp_path):
    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    _make_tone(a_path, 220.0, seconds=4.0)
    _make_tone(b_path, 330.0, seconds=4.0)
    with pytest.raises(ValueError):
        fuse(a_path, b_path, style="nope", sr=22_050)


def test_fuse_analysis_cache(tmp_path):
    from songfuse import analysis as mod

    a_path = tmp_path / "a.wav"
    _make_tone(a_path, 220.0, seconds=3.0)
    mod.clear_cache()
    first = mod.analyze(a_path)
    second = mod.analyze(a_path)
    assert first is second  # cache hit returns the exact same object
