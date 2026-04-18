"""Stereo-aware audio I/O helpers.

Internally audio is (channels, samples), float32. Mono is (1, samples)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_SR = 44_100


def load(path: str | Path, sr: int = DEFAULT_SR, mono: bool = False) -> tuple[np.ndarray, int]:
    import librosa

    y, file_sr = librosa.load(str(path), sr=sr, mono=mono)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    return y.astype(np.float32, copy=False), int(file_sr)


def save(path: str | Path, audio: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.ndim == 1:
        data = audio
    elif audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
        data = audio.T
    else:
        data = audio
    # PCM targets require [-1, 1]; float subtypes can tolerate slight overs but it's safer to clip.
    data = np.clip(data, -1.0, 1.0)
    sf.write(str(path), data, sr, subtype=subtype)


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=0)


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=0)
    if audio.shape[0] == 1:
        return np.concatenate([audio, audio], axis=0)
    return audio


def trim_silence(audio: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    """Strip leading/trailing silence. Uses librosa's energy-based trim."""
    import librosa

    if audio.size == 0:
        return audio
    mono = to_mono(audio)
    _, interval = librosa.effects.trim(mono, top_db=top_db)
    start, end = int(interval[0]), int(interval[1])
    return audio[..., start:end] if end > start else audio
