"""Music Information Retrieval: tempo, beats, downbeats, key, structural segments.

Intentionally MVP: librosa-based. Swap in madmom/Essentia/EDMFormer for production."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from songfuse import camelot
from songfuse.audio_io import DEFAULT_SR, load, to_mono

# Krumhansl-Schmuckler key profiles (experimentally-derived tonal hierarchies).
_KRUM_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KRUM_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


@dataclass
class Segment:
    start: float
    end: float
    kind: str  # 'intro' | 'outro' | 'body' | 'low_energy' | 'high_energy'

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TrackAnalysis:
    path: str
    sr: int
    duration: float
    bpm: float
    beats: np.ndarray          # beat times in seconds
    downbeats: np.ndarray      # estimated downbeat times in seconds
    key_pc: int                # 0..11 (C..B)
    key_mode: str              # 'major' | 'minor'
    key_confidence: float
    segments: list[Segment] = field(default_factory=list)

    @property
    def key_name(self) -> str:
        return camelot.key_name(self.key_pc, self.key_mode)

    @property
    def camelot(self) -> str:
        return str(camelot.from_key(self.key_pc, self.key_mode))

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sr": self.sr,
            "duration": round(self.duration, 3),
            "bpm": round(self.bpm, 2),
            "key": self.key_name,
            "camelot": self.camelot,
            "key_confidence": round(self.key_confidence, 3),
            "n_beats": int(self.beats.size),
            "n_downbeats": int(self.downbeats.size),
            "segments": [
                {"start": round(s.start, 2), "end": round(s.end, 2), "kind": s.kind}
                for s in self.segments
            ],
        }


def _detect_key(chroma: np.ndarray) -> tuple[int, str, float]:
    """Krumhansl correlation across 24 key templates."""
    mean_chroma = chroma.mean(axis=1)
    if not np.isfinite(mean_chroma).all() or mean_chroma.sum() < 1e-6:
        return 0, "major", 0.0  # undefined key for silent / degenerate input
    best_score = -np.inf
    best_pc, best_mode = 0, "major"
    for pc in range(12):
        for profile, mode in ((_KRUM_MAJOR, "major"), (_KRUM_MINOR, "minor")):
            rotated = np.roll(profile, pc)
            with np.errstate(invalid="ignore"):
                score = float(np.corrcoef(mean_chroma, rotated)[0, 1])
            if not np.isfinite(score):
                continue
            if score > best_score:
                best_score = score
                best_pc, best_mode = pc, mode
    if not np.isfinite(best_score):
        return 0, "major", 0.0
    return best_pc, best_mode, max(0.0, min(1.0, (best_score + 1) / 2))


def _estimate_downbeats(beats: np.ndarray, onset_env: np.ndarray | None = None,
                        sr: int | None = None, hop: int | None = None) -> np.ndarray:
    """Approximate downbeats as every 4th beat, choosing the phase (0..3) whose
    beat positions have the highest mean onset strength. Falls back to phase=0."""
    if beats.size < 4:
        return beats
    if onset_env is None or sr is None or hop is None:
        return beats[::4]
    frames = np.clip((beats * sr / hop).astype(int), 0, onset_env.size - 1)
    strengths = onset_env[frames]
    best_phase, best_score = 0, -np.inf
    for phase in range(4):
        score = float(strengths[phase::4].mean()) if strengths[phase::4].size else 0.0
        if score > best_score:
            best_phase, best_score = phase, score
    return beats[best_phase::4]


def _segment(y: np.ndarray, sr: int, duration: float) -> list[Segment]:
    """Coarse energy-based segmentation: intro / body / outro + peak markers.

    This is intentionally simple — swap with MSAF or EDMFormer for production."""
    import librosa

    hop = 1024
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop)
    if rms.size < 4:
        return [Segment(0.0, duration, "body")]

    normalized = rms / (rms.max() + 1e-8)
    low_threshold = 0.35
    intro_end = float(times[np.argmax(normalized > low_threshold)]) if np.any(normalized > low_threshold) else 0.0
    outro_start = float(
        times[len(times) - 1 - np.argmax(normalized[::-1] > low_threshold)]
    ) if np.any(normalized > low_threshold) else duration
    segments = []
    if intro_end > 1.0:
        segments.append(Segment(0.0, intro_end, "intro"))
    body_start = intro_end
    body_end = max(outro_start, body_start + 1.0)
    segments.append(Segment(body_start, body_end, "body"))
    if duration - body_end > 1.0:
        segments.append(Segment(body_end, duration, "outro"))
    return segments


_ANALYSIS_CACHE: dict[tuple[str, int, int], "TrackAnalysis"] = {}


def _cache_key(path: Path, sr: int) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()), sr, stat.st_mtime_ns)


def clear_cache() -> None:
    _ANALYSIS_CACHE.clear()


def analyze(path: str | Path, sr: int = DEFAULT_SR, *, use_cache: bool = True) -> TrackAnalysis:
    """Run the full annotation pipeline on a single track. Results are memoized
    by (resolved path, sample rate, mtime) — repeated calls are free."""
    import librosa

    resolved = Path(path)
    key = _cache_key(resolved, sr) if use_cache and resolved.exists() else None
    if key is not None and key in _ANALYSIS_CACHE:
        return _ANALYSIS_CACHE[key]

    audio, sr = load(path, sr=sr)
    mono = to_mono(audio)
    if mono.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    duration = mono.size / sr

    hop = 512
    onset_env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop, start_bpm=120, units="frames"
    )
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    bpm = float(np.atleast_1d(tempo)[0])
    if not np.isfinite(bpm) or bpm <= 0:
        # Degenerate: fall back to a neutral tempo so downstream math doesn't explode.
        bpm = 120.0

    harmonic = librosa.effects.harmonic(mono)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, bins_per_octave=36)
    key_pc, key_mode, key_conf = _detect_key(chroma)

    segments = _segment(mono, sr, duration)
    downbeats = _estimate_downbeats(beats, onset_env=onset_env, sr=sr, hop=hop)

    result = TrackAnalysis(
        path=str(path),
        sr=sr,
        duration=duration,
        bpm=bpm,
        beats=beats,
        downbeats=downbeats,
        key_pc=key_pc,
        key_mode=key_mode,
        key_confidence=key_conf,
        segments=segments,
    )
    if key is not None:
        _ANALYSIS_CACHE[key] = result
    return result
