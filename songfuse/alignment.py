"""Temporal & tonal alignment: tempo matching and pitch shifting."""

from __future__ import annotations

import numpy as np


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Rate > 1 speeds up. Preserves pitch. Channel-by-channel."""
    import librosa

    if abs(rate - 1.0) < 1e-4:
        return audio
    if audio.ndim == 1:
        return librosa.effects.time_stretch(audio, rate=rate)
    stretched = [librosa.effects.time_stretch(ch, rate=rate) for ch in audio]
    min_len = min(ch.size for ch in stretched)
    return np.stack([ch[:min_len] for ch in stretched], axis=0)


def pitch_shift(audio: np.ndarray, sr: int, n_semitones: float) -> np.ndarray:
    import librosa

    if abs(n_semitones) < 1e-4:
        return audio
    if audio.ndim == 1:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_semitones)
    return np.stack(
        [librosa.effects.pitch_shift(ch, sr=sr, n_steps=n_semitones) for ch in audio],
        axis=0,
    )


def pick_target_bpm(bpm_a: float, bpm_b: float, strategy: str = "b") -> float:
    if strategy == "a":
        return bpm_a
    if strategy == "b":
        return bpm_b
    if strategy == "avg":
        return (bpm_a + bpm_b) / 2.0
    raise ValueError(f"Unknown tempo strategy: {strategy}")


def tempo_variance_ok(bpm_a: float, bpm_b: float, max_ratio: float = 0.06) -> bool:
    """Return True if the tempos are close enough to DSP-match (default ±6%)."""
    ratio = abs(bpm_a - bpm_b) / max(bpm_a, bpm_b)
    return ratio <= max_ratio


def align_to_downbeat(audio: np.ndarray, sr: int, downbeats: np.ndarray,
                      from_time: float) -> tuple[np.ndarray, float]:
    """Trim `audio` so it starts on the nearest downbeat ≥ from_time. Returns audio + actual start."""
    if downbeats.size == 0:
        start = int(from_time * sr)
        return audio[..., start:], from_time
    candidates = downbeats[downbeats >= from_time]
    start_t = float(candidates[0]) if candidates.size else float(downbeats[-1])
    start = int(start_t * sr)
    return audio[..., start:], start_t
