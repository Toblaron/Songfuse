"""Deterministic DSP sanity checks. No audio files needed."""

from __future__ import annotations

import numpy as np
import pytest

from songfuse import camelot
from songfuse.alignment import pick_target_bpm, tempo_variance_ok
from songfuse.transition import (
    bass_swap_crossfade,
    constant_power_crossfade,
    crossfade,
    equal_power_curves,
    linear_crossfade,
)


def test_equal_power_curves_preserve_power():
    fa, fb = equal_power_curves(1024)
    power = fa ** 2 + fb ** 2
    assert np.allclose(power, 1.0, atol=1e-5)


def test_constant_power_crossfade_no_midpoint_dip():
    sr = 44_100
    n = sr
    t = np.linspace(0, 1, n, endpoint=False, dtype=np.float32)
    a = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    b = np.sin(2 * np.pi * 330 * t).astype(np.float32)
    mixed = constant_power_crossfade(a, b)
    midpoint = mixed[n // 2 - 1000: n // 2 + 1000]
    start = mixed[:2000]
    end = mixed[-2000:]
    # Midpoint RMS should stay within ~30% of the endpoints for constant-power fades.
    assert np.sqrt((midpoint ** 2).mean()) > 0.6 * np.sqrt((start ** 2).mean())
    assert np.sqrt((midpoint ** 2).mean()) > 0.6 * np.sqrt((end ** 2).mean())


def test_linear_crossfade_has_midpoint_dip():
    sr = 44_100
    n = sr
    t = np.linspace(0, 1, n, endpoint=False, dtype=np.float32)
    a = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    b = np.sin(2 * np.pi * 330 * t).astype(np.float32)
    mixed = linear_crossfade(a, b)
    midpoint_rms = np.sqrt((mixed[n // 2 - 1000: n // 2 + 1000] ** 2).mean())
    endpoint_rms = np.sqrt((mixed[:2000] ** 2).mean())
    # Linear crossfade is *expected* to dip at the midpoint.
    assert midpoint_rms < endpoint_rms


def test_bass_swap_is_bounded():
    sr = 44_100
    n = 2 * sr
    t = np.linspace(0, n / sr, n, endpoint=False, dtype=np.float32)
    a = (0.5 * np.sin(2 * np.pi * 80 * t) + 0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    b = (0.5 * np.sin(2 * np.pi * 90 * t) + 0.3 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)
    mixed = bass_swap_crossfade(a, b, sr)
    assert np.isfinite(mixed).all()
    assert np.max(np.abs(mixed)) < 1.5


def test_crossfade_dispatch_rejects_unknown_style():
    with pytest.raises(ValueError):
        crossfade("not_a_style", np.zeros(10, dtype=np.float32),
                  np.zeros(10, dtype=np.float32), 44_100)


def test_camelot_compatibility_rules():
    c_major = camelot.from_key(0, "major")
    a_minor = camelot.from_key(9, "minor")  # relative minor
    g_major = camelot.from_key(7, "major")  # +1 energy boost on the wheel
    f_sharp_major = camelot.from_key(6, "major")  # 4 wheel-steps away, incompatible

    assert camelot.is_compatible(c_major, a_minor)
    assert camelot.is_compatible(c_major, g_major)
    assert not camelot.is_compatible(c_major, f_sharp_major)


def test_camelot_small_shift_finds_compatibility():
    shift = camelot.semitone_shift_to_match(6, "major", 0, "major")
    # F# major -> C major is incompatible; shifting by ±1 semitone reaches a compatible key.
    assert shift in {-1, 1}


def test_tempo_helpers():
    assert tempo_variance_ok(128.0, 130.0)
    assert not tempo_variance_ok(90.0, 130.0)
    assert pick_target_bpm(120.0, 128.0, "b") == 128.0
    assert pick_target_bpm(120.0, 128.0, "avg") == 124.0
