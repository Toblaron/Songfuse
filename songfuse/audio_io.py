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


def save(path: str | Path, audio: np.ndarray, sr: int) -> None:
    if audio.ndim == 1:
        data = audio
    elif audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
        data = audio.T
    else:
        data = audio
    sf.write(str(path), data, sr, subtype="PCM_16")


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
