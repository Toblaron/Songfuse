"""EBU R128 loudness normalization to a target LUFS.

Uses pyloudnorm (K-weighted meter). Falls back to RMS matching if unavailable."""

from __future__ import annotations

import numpy as np


def _as_pyln(audio: np.ndarray) -> np.ndarray:
    """pyloudnorm wants (samples,) or (samples, channels)."""
    if audio.ndim == 1:
        return audio
    return audio.T


def measure_lufs(audio: np.ndarray, sr: int) -> float:
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(_as_pyln(audio)))
    except Exception:  # noqa: BLE001
        rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
        return 20.0 * np.log10(rms + 1e-12) - 0.691  # crude K-weighting proxy


def normalize(audio: np.ndarray, sr: int, target_lufs: float = -14.0,
              peak_ceiling: float = -1.0) -> np.ndarray:
    """Apply integrated-loudness gain to hit `target_lufs`, with a true-peak ceiling."""
    current = measure_lufs(audio, sr)
    if not np.isfinite(current):
        return audio
    gain_db = target_lufs - current
    gain = 10 ** (gain_db / 20.0)
    out = audio * gain
    peak = float(np.max(np.abs(out)) + 1e-12)
    peak_limit = 10 ** (peak_ceiling / 20.0)
    if peak > peak_limit:
        out *= peak_limit / peak
    return out.astype(audio.dtype, copy=False)
