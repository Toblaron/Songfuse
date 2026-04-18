"""Top-level orchestrator: fuse two tracks with a seamless transition."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from songfuse import camelot
from songfuse.alignment import (
    align_to_downbeat,
    pick_target_bpm,
    pitch_shift,
    tempo_variance_ok,
    time_stretch,
)
from songfuse.analysis import TrackAnalysis, analyze
from songfuse.audio_io import DEFAULT_SR, ensure_stereo, load, save, to_mono
from songfuse.loudness import normalize
from songfuse.transition import crossfade


@dataclass
class FusePlan:
    target_bpm: float
    bars: int
    style: str
    tempo_variance_ok: bool
    semitone_shift_a: int
    switch_time_a: float
    switch_time_b: float
    crossfade_seconds: float
    ai_bridge_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class FuseResult:
    audio: np.ndarray
    sr: int
    plan: FusePlan
    analysis_a: TrackAnalysis
    analysis_b: TrackAnalysis

    def to_summary(self) -> dict:
        return {
            "plan": asdict(self.plan),
            "track_a": self.analysis_a.to_dict(),
            "track_b": self.analysis_b.to_dict(),
            "output_seconds": round(self.audio.shape[-1] / self.sr, 3),
        }


def _bars_to_samples(bars: int, bpm: float, sr: int, beats_per_bar: int = 4) -> int:
    seconds = bars * beats_per_bar * (60.0 / bpm)
    return int(seconds * sr)


def _pick_outro_start(a: TrackAnalysis, crossfade_seconds: float) -> float:
    """Anchor the crossfade on a downbeat near the end of track A, preferring the outro."""
    target = a.duration - crossfade_seconds
    outro_segments = [s for s in a.segments if s.kind == "outro"]
    if outro_segments:
        target = max(outro_segments[0].start, target - 1.0)
    target = max(0.0, min(target, a.duration - crossfade_seconds))
    if a.downbeats.size == 0:
        return target
    candidates = a.downbeats[a.downbeats <= target]
    return float(candidates[-1]) if candidates.size else float(a.downbeats[0])


def _pick_intro_start(b: TrackAnalysis) -> float:
    """Start from the first downbeat past the intro (or first downbeat if no intro tagged)."""
    intro_segments = [s for s in b.segments if s.kind == "intro"]
    target = intro_segments[0].end if intro_segments else 0.0
    if b.downbeats.size == 0:
        return target
    candidates = b.downbeats[b.downbeats >= target]
    return float(candidates[0]) if candidates.size else 0.0


def fuse(
    path_a: str | Path,
    path_b: str | Path,
    *,
    out: str | Path | None = None,
    style: str = "eq",
    bars: int = 16,
    match_key: bool = True,
    tempo_strategy: str = "b",
    target_bpm: float | None = None,
    lufs_target: float = -14.0,
    ai_bridge_seconds: float = 0.0,
    ai_bridge_prompt: str | None = None,
    sr: int = DEFAULT_SR,
) -> FuseResult:
    """Fuse two songs with a beat-matched, harmonically aware transition.

    Pipeline: analyze → LUFS-normalize → pitch-shift A if needed → tempo-match to
    `target_bpm` → crossfade on aligned downbeats → (optional) splice MusicGen bridge → save.
    """
    a_analysis = analyze(path_a, sr=sr)
    b_analysis = analyze(path_b, sr=sr)

    audio_a, _ = load(path_a, sr=sr)
    audio_b, _ = load(path_b, sr=sr)
    audio_a = ensure_stereo(audio_a)
    audio_b = ensure_stereo(audio_b)

    audio_a = normalize(audio_a, sr, target_lufs=lufs_target)
    audio_b = normalize(audio_b, sr, target_lufs=lufs_target)

    chosen_bpm = target_bpm or pick_target_bpm(a_analysis.bpm, b_analysis.bpm, tempo_strategy)
    variance_ok = tempo_variance_ok(a_analysis.bpm, b_analysis.bpm)
    notes: list[str] = []
    if not variance_ok:
        notes.append(
            f"Tempo variance {abs(a_analysis.bpm - b_analysis.bpm):.1f} BPM is >6%; "
            "consider --ai-bridge-seconds for a generative bridge."
        )

    # Harmonic alignment: shift A by the smallest step that achieves Camelot compatibility.
    semitone_shift = 0
    if match_key:
        semitone_shift = camelot.semitone_shift_to_match(
            a_analysis.key_pc, a_analysis.key_mode,
            b_analysis.key_pc, b_analysis.key_mode,
        )
        if semitone_shift != 0:
            audio_a = pitch_shift(audio_a, sr, semitone_shift)
            notes.append(
                f"Pitch-shifted track A by {semitone_shift:+d} semitone(s) for harmonic compatibility."
            )

    # Tempo match: stretch each to chosen_bpm.
    rate_a = a_analysis.bpm / chosen_bpm
    rate_b = b_analysis.bpm / chosen_bpm
    stretched_a = time_stretch(audio_a, rate_a)
    stretched_b = time_stretch(audio_b, rate_b)

    # Scale downbeat times by the stretch ratio (inverse of rate).
    a_downbeats_scaled = a_analysis.downbeats / rate_a if a_analysis.downbeats.size else a_analysis.downbeats
    b_downbeats_scaled = b_analysis.downbeats / rate_b if b_analysis.downbeats.size else b_analysis.downbeats

    # Switch points on the stretched timeline.
    crossfade_seconds = bars * 4 * (60.0 / chosen_bpm)
    a_scaled_duration = stretched_a.shape[-1] / sr
    b_scaled_duration = stretched_b.shape[-1] / sr

    a_scaled = _rewrap_analysis(a_analysis, a_downbeats_scaled, a_scaled_duration, rate_a)
    b_scaled = _rewrap_analysis(b_analysis, b_downbeats_scaled, b_scaled_duration, rate_b)
    switch_a = _pick_outro_start(a_scaled, crossfade_seconds)
    switch_b = _pick_intro_start(b_scaled)

    # Slice segments.
    crossfade_samples = int(crossfade_seconds * sr)
    start_a_samples = int(switch_a * sr)
    end_a_samples = min(start_a_samples + crossfade_samples, stretched_a.shape[-1])
    body_a = stretched_a[..., :start_a_samples]
    tail_a = stretched_a[..., start_a_samples:end_a_samples]

    start_b_samples = int(switch_b * sr)
    head_b_end = min(start_b_samples + crossfade_samples, stretched_b.shape[-1])
    head_b = stretched_b[..., start_b_samples:head_b_end]
    body_b = stretched_b[..., head_b_end:]

    # Pad to equal overlap length.
    n = min(tail_a.shape[-1], head_b.shape[-1])
    tail_a = tail_a[..., :n]
    head_b = head_b[..., :n]

    # Optional generative bridge inserted before the crossfade.
    bridge: np.ndarray | None = None
    if ai_bridge_seconds > 0:
        from songfuse.ai_bridge import generate_bridge

        bridge = generate_bridge(
            tail_a, head_b, sr,
            duration=ai_bridge_seconds,
            prompt=ai_bridge_prompt,
        )
        bridge = ensure_stereo(bridge)
        notes.append(f"Inserted {ai_bridge_seconds:.1f}s MusicGen bridge.")

    # Crossfade the overlap.
    mixed_overlap = crossfade(style, tail_a, head_b, sr)

    # Compose output.
    pieces = [body_a]
    if bridge is not None:
        bridge_fade_samples = min(int(0.2 * sr), bridge.shape[-1] // 4, body_a.shape[-1] // 4 if body_a.shape[-1] else 0)
        if bridge_fade_samples > 0:
            pieces[-1] = _crossfade_tails(body_a, bridge, bridge_fade_samples)
        else:
            pieces.append(bridge)
    pieces.append(mixed_overlap)
    pieces.append(body_b)
    output = np.concatenate([p for p in pieces if p.size], axis=-1)

    # Final true-peak guard after summation.
    peak = float(np.max(np.abs(output)) + 1e-9)
    if peak > 0.99:
        output = output * (0.99 / peak)

    plan = FusePlan(
        target_bpm=float(chosen_bpm),
        bars=bars,
        style=style,
        tempo_variance_ok=variance_ok,
        semitone_shift_a=semitone_shift,
        switch_time_a=round(switch_a, 3),
        switch_time_b=round(switch_b, 3),
        crossfade_seconds=round(crossfade_seconds, 3),
        ai_bridge_seconds=float(ai_bridge_seconds),
        notes=notes,
    )

    result = FuseResult(audio=output, sr=sr, plan=plan,
                        analysis_a=a_analysis, analysis_b=b_analysis)

    if out is not None:
        save(out, output, sr)
    return result


def _rewrap_analysis(src: TrackAnalysis, downbeats: np.ndarray, duration: float,
                     stretch_rate: float) -> TrackAnalysis:
    """Return a lightweight TrackAnalysis copy with timings rescaled by stretching."""
    rescaled = TrackAnalysis(
        path=src.path,
        sr=src.sr,
        duration=duration,
        bpm=src.bpm / stretch_rate,
        beats=src.beats / stretch_rate,
        downbeats=downbeats,
        key_pc=src.key_pc,
        key_mode=src.key_mode,
        key_confidence=src.key_confidence,
        segments=[type(s)(start=s.start / stretch_rate, end=s.end / stretch_rate, kind=s.kind)
                  for s in src.segments],
    )
    return rescaled


def _crossfade_tails(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Short equal-power tail-of-a / head-of-b glue."""
    from songfuse.transition import constant_power_crossfade

    if n <= 0:
        return np.concatenate([a, b], axis=-1)
    tail_a = a[..., -n:]
    head_b = b[..., :n]
    body_a = a[..., :-n]
    body_b = b[..., n:]
    glue = constant_power_crossfade(tail_a, head_b)
    return np.concatenate([body_a, glue, body_b], axis=-1)
