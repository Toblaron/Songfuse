"""Unit tests for the AI bridge surface.

These tests never load MusicGen — they either exercise pure helpers or
monkey-patch `generate_transition` to verify that `fuse()` wires `ai_mode`
and `ai_model` through correctly."""

from __future__ import annotations

import numpy as np
import pytest

from songfuse.ai_bridge import (
    VALID_MODES,
    AIBridgeUnavailable,
    _ensure_mono,
    _ensure_stereo,
    generate_transition,
)


def test_ensure_mono_from_stereo():
    stereo = np.stack([np.ones(100, dtype=np.float32),
                       -np.ones(100, dtype=np.float32)], axis=0)
    mono = _ensure_mono(stereo)
    assert mono.shape == (1, 100)
    assert np.allclose(mono, 0.0)


def test_ensure_mono_from_1d():
    one_d = np.arange(10, dtype=np.float32)
    mono = _ensure_mono(one_d)
    assert mono.shape == (1, 10)


def test_ensure_stereo_from_mono():
    mono = np.ones((1, 50), dtype=np.float32)
    stereo = _ensure_stereo(mono)
    assert stereo.shape == (2, 50)
    assert np.array_equal(stereo[0], stereo[1])


def test_ensure_stereo_from_1d():
    one_d = np.ones(50, dtype=np.float32)
    stereo = _ensure_stereo(one_d)
    assert stereo.shape == (2, 50)


def test_generate_transition_rejects_bad_mode():
    with pytest.raises(ValueError):
        generate_transition(
            np.zeros((2, 100), dtype=np.float32),
            np.zeros((2, 100), dtype=np.float32),
            sr=44_100, mode="not_a_mode",
        )


def test_generate_transition_rejects_nonpositive_duration():
    with pytest.raises(ValueError):
        generate_transition(
            np.zeros((2, 100), dtype=np.float32),
            np.zeros((2, 100), dtype=np.float32),
            sr=44_100, duration=0,
        )


def test_valid_modes_contains_expected_entries():
    assert set(VALID_MODES) == {"bidirectional", "forward", "backward"}


def test_generate_transition_raises_when_audiocraft_missing(monkeypatch):
    """If audiocraft can't be imported, we should surface a clear AIBridgeUnavailable."""
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("audiocraft"):
            raise ImportError("audiocraft not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(AIBridgeUnavailable):
        generate_transition(
            np.zeros((2, 100), dtype=np.float32),
            np.zeros((2, 100), dtype=np.float32),
            sr=44_100, duration=1.0,
        )


def test_fuse_forwards_ai_mode_and_model(tmp_path, monkeypatch):
    """fuse() should call generate_transition with the mode+model+prompt+seed we passed."""
    librosa = pytest.importorskip("librosa")  # noqa: F841

    from songfuse.audio_io import save
    from songfuse.fuse import fuse

    sr = 22_050
    seconds = 6.0
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    click = np.zeros_like(t)
    for i in range(0, click.size, int(sr * 0.5)):
        click[i: i + int(sr * 0.02)] += 0.8
    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    save(a_path, 0.5 * np.sin(2 * np.pi * 220 * t) + click, sr)
    save(b_path, 0.5 * np.sin(2 * np.pi * 330 * t) + click, sr)

    captured: dict = {}

    def fake_generate_transition(tail_a, head_b, working_sr, *, duration, mode,
                                 model_name, prompt, seed):
        captured["mode"] = mode
        captured["model_name"] = model_name
        captured["prompt"] = prompt
        captured["seed"] = seed
        captured["duration"] = duration
        captured["sr"] = working_sr
        return np.zeros((2, int(duration * working_sr)), dtype=np.float32)

    import songfuse.ai_bridge as ai_module

    monkeypatch.setattr(ai_module, "generate_transition", fake_generate_transition)

    fuse(
        a_path, b_path,
        style="equal_power", bars=1, match_key=False,
        sr=sr, lufs_target=-20.0,
        ai_bridge_seconds=1.0,
        ai_bridge_mode="forward",
        ai_bridge_model="facebook/musicgen-small",
        ai_bridge_prompt="warm pad sweep",
        ai_bridge_seed=7,
    )

    assert captured["mode"] == "forward"
    assert captured["model_name"] == "facebook/musicgen-small"
    assert captured["prompt"] == "warm pad sweep"
    assert captured["seed"] == 7
    assert captured["duration"] == 1.0
    assert captured["sr"] == sr
