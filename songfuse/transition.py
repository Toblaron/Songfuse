"""Crossfade curves and frequency-split (bass-swap) transitions."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def equal_power_curves(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Constant-power (sin/cos) crossfade: preserves summed RMS across the overlap."""
    t = np.linspace(0.0, np.pi / 2.0, length, dtype=np.float32)
    return np.cos(t), np.sin(t)


def linear_curves(length: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return 1.0 - t, t


def _broadcast(curve: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return curve if ref.ndim == 1 else curve[np.newaxis, :]


def constant_power_crossfade(tail_a: np.ndarray, head_b: np.ndarray) -> np.ndarray:
    """Overlap both segments fully; equal-power sum."""
    n = min(tail_a.shape[-1], head_b.shape[-1])
    a = tail_a[..., :n]
    b = head_b[..., :n]
    fa, fb = equal_power_curves(n)
    return a * _broadcast(fa, a) + b * _broadcast(fb, b)


def _split_low_high(audio: np.ndarray, sr: int, crossover_hz: float = 180.0) -> tuple[np.ndarray, np.ndarray]:
    sos_low = butter(4, crossover_hz, btype="low", fs=sr, output="sos")
    sos_high = butter(4, crossover_hz, btype="high", fs=sr, output="sos")

    def _apply(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return sosfiltfilt(sos, x).astype(np.float32, copy=False)
        return np.stack([sosfiltfilt(sos, ch) for ch in x], axis=0).astype(np.float32, copy=False)

    return _apply(sos_low, audio), _apply(sos_high, audio)


def bass_swap_crossfade(tail_a: np.ndarray, head_b: np.ndarray, sr: int,
                        swap_point: float = 0.25, crossover_hz: float = 180.0) -> np.ndarray:
    """Swap the low band early on the overlap (classic DJ bass swap), equal-power the highs.

    swap_point ∈ (0, 1): fraction of overlap at which lows fully swap. Smaller = earlier swap.
    """
    n = min(tail_a.shape[-1], head_b.shape[-1])
    a = tail_a[..., :n]
    b = head_b[..., :n]

    a_low, a_high = _split_low_high(a, sr, crossover_hz)
    b_low, b_high = _split_low_high(b, sr, crossover_hz)

    # High band: standard equal-power fade across the whole overlap.
    fa_high, fb_high = equal_power_curves(n)

    # Low band: hard-ish swap centred at `swap_point`. Smooth sigmoid for click-free switch.
    idx = np.arange(n, dtype=np.float32) / max(n - 1, 1)
    sharpness = 12.0
    fb_low = 1.0 / (1.0 + np.exp(-sharpness * (idx - swap_point)))
    fa_low = 1.0 - fb_low
    fa_low = fa_low.astype(np.float32)
    fb_low = fb_low.astype(np.float32)

    return (
        a_high * _broadcast(fa_high, a_high)
        + b_high * _broadcast(fb_high, b_high)
        + a_low * _broadcast(fa_low, a_low)
        + b_low * _broadcast(fb_low, b_low)
    )


def linear_crossfade(tail_a: np.ndarray, head_b: np.ndarray) -> np.ndarray:
    n = min(tail_a.shape[-1], head_b.shape[-1])
    a = tail_a[..., :n]
    b = head_b[..., :n]
    fa, fb = linear_curves(n)
    return a * _broadcast(fa, a) + b * _broadcast(fb, b)


CROSSFADE_FNS = {
    "linear": lambda a, b, sr: linear_crossfade(a, b),
    "equal_power": lambda a, b, sr: constant_power_crossfade(a, b),
    "eq": lambda a, b, sr: bass_swap_crossfade(a, b, sr),
    "bass_swap": lambda a, b, sr: bass_swap_crossfade(a, b, sr),
}


def crossfade(style: str, tail_a: np.ndarray, head_b: np.ndarray, sr: int) -> np.ndarray:
    fn = CROSSFADE_FNS.get(style)
    if fn is None:
        raise ValueError(f"Unknown crossfade style: {style!r}. Options: {sorted(CROSSFADE_FNS)}")
    return fn(tail_a, head_b, sr)
