"""Top-level orchestrator: fuse two tracks with a seamless transition."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from songfuse import camelot
from songfuse.alignment import (
    pick_target_bpm,
    pitch_shift,
    tempo_variance_ok,
    time_stretch,
)
from songfuse.analysis import TrackAnalysis, analyze
from songfuse.audio_io import DEFAULT_SR, ensure_stereo, load, save
from songfuse.loudness import normalize
from songfuse.transition import constant_power_crossfade, crossfade

VALID_STYLES = ("linear", "equal_power", "eq", "bass_swap", "stems")


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
    skip_head_a: float = 0.0
    skip_tail_a: float = 0.0
    skip_head_b: float = 0.0
    skip_tail_b: float = 0.0
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


def _bars_to_seconds(bars: int, bpm: float, beats_per_bar: int = 4) -> float:
    return bars * beats_per_bar * (60.0 / max(bpm, 1e-6))


def _pick_outro_start(a: TrackAnalysis, crossfade_seconds: float) -> float:
    """Anchor the crossfade on a downbeat near the end of track A, preferring the outro."""
    target = max(0.0, a.duration - crossfade_seconds)
    outro_segments = [s for s in a.segments if s.kind == "outro"]
    if outro_segments:
        target = max(outro_segments[0].start, target - 1.0)
    target = max(0.0, min(target, max(a.duration - crossfade_seconds, 0.0)))
    if a.downbeats.size == 0:
        return target
    candidates = a.downbeats[a.downbeats <= target]
    return float(candidates[-1]) if candidates.size else 0.0


def _pick_intro_start(b: TrackAnalysis) -> float:
    """Start from the first downbeat past the intro (or first downbeat if no intro tagged)."""
    intro_segments = [s for s in b.segments if s.kind == "intro"]
    target = intro_segments[0].end if intro_segments else 0.0
    if b.downbeats.size == 0:
        return target
    candidates = b.downbeats[b.downbeats >= target]
    return float(candidates[0]) if candidates.size else 0.0


def _apply_skips(audio: np.ndarray, sr: int, skip_head: float, skip_tail: float) -> np.ndarray:
    total = audio.shape[-1]
    start = max(0, int(skip_head * sr))
    end = total - max(0, int(skip_tail * sr))
    if end <= start:
        return audio[..., :0]
    return audio[..., start:end]


def _clamp_bars(bars: int, bpm: float, dur_a: float, dur_b: float, min_body: float = 1.0) -> int:
    """Shrink the crossfade so both tracks keep at least `min_body` seconds of solo playback."""
    max_seconds = max(1.0, min(dur_a, dur_b) - min_body)
    max_bars = int(max(1, max_seconds / (4 * 60.0 / max(bpm, 1e-6))))
    return max(1, min(bars, max_bars))


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
    skip_head_a: float = 0.0,
    skip_tail_a: float = 0.0,
    skip_head_b: float = 0.0,
    skip_tail_b: float = 0.0,
    ai_bridge_seconds: float = 0.0,
    ai_bridge_prompt: str | None = None,
    ai_bridge_seed: int | None = None,
    sr: int = DEFAULT_SR,
) -> FuseResult:
    """Fuse two songs with a beat-matched, harmonically aware transition."""
    if style not in VALID_STYLES:
        raise ValueError(f"Unknown style {style!r}. Valid: {VALID_STYLES}")

    a_analysis = analyze(path_a, sr=sr)
    b_analysis = analyze(path_b, sr=sr)

    audio_a, _ = load(path_a, sr=sr)
    audio_b, _ = load(path_b, sr=sr)
    audio_a = ensure_stereo(audio_a)
    audio_b = ensure_stereo(audio_b)

    # User-requested pre-trim (e.g. skip a long intro before DJing into the track).
    audio_a = _apply_skips(audio_a, sr, 0.0, skip_tail_a)[..., int(skip_head_a * sr):]
    audio_b = _apply_skips(audio_b, sr, 0.0, skip_tail_b)[..., int(skip_head_b * sr):]

    audio_a = normalize(audio_a, sr, target_lufs=lufs_target)
    audio_b = normalize(audio_b, sr, target_lufs=lufs_target)

    chosen_bpm = target_bpm or pick_target_bpm(a_analysis.bpm, b_analysis.bpm, tempo_strategy)
    if not np.isfinite(chosen_bpm) or chosen_bpm <= 0:
        chosen_bpm = 120.0
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

    # librosa.time_stretch uses rate>1 to SHORTEN (speed up); rate<1 LENGTHENS (slow down).
    # To reach `chosen_bpm` from `original_bpm`, rate = chosen_bpm / original_bpm.
    rate_a = chosen_bpm / max(a_analysis.bpm, 1e-6)
    rate_b = chosen_bpm / max(b_analysis.bpm, 1e-6)
    stretched_a = time_stretch(audio_a, rate_a)
    stretched_b = time_stretch(audio_b, rate_b)

    # Apply the same stretch to the annotation so our switch points land on downbeats.
    a_downbeats_scaled = a_analysis.downbeats / rate_a if a_analysis.downbeats.size else a_analysis.downbeats
    b_downbeats_scaled = b_analysis.downbeats / rate_b if b_analysis.downbeats.size else b_analysis.downbeats
    # Account for the user-skipped intro on B so downbeat times match the trimmed audio.
    if skip_head_b > 0 and b_downbeats_scaled.size:
        b_downbeats_scaled = b_downbeats_scaled - skip_head_b
        b_downbeats_scaled = b_downbeats_scaled[b_downbeats_scaled >= 0]
    if skip_head_a > 0 and a_downbeats_scaled.size:
        a_downbeats_scaled = a_downbeats_scaled - skip_head_a
        a_downbeats_scaled = a_downbeats_scaled[a_downbeats_scaled >= 0]

    a_scaled_duration = stretched_a.shape[-1] / sr
    b_scaled_duration = stretched_b.shape[-1] / sr

    # Clamp bars so the crossfade fits inside both tracks with at least a little solo time.
    effective_bars = _clamp_bars(bars, chosen_bpm, a_scaled_duration, b_scaled_duration)
    if effective_bars != bars:
        notes.append(f"Clamped crossfade from {bars} bars to {effective_bars} to fit track lengths.")
    crossfade_seconds = _bars_to_seconds(effective_bars, chosen_bpm)

    a_scaled = _rewrap_analysis(a_analysis, a_downbeats_scaled, a_scaled_duration, rate_a)
    b_scaled = _rewrap_analysis(b_analysis, b_downbeats_scaled, b_scaled_duration, rate_b)
    switch_a = _pick_outro_start(a_scaled, crossfade_seconds)
    switch_b = _pick_intro_start(b_scaled)

    crossfade_samples = int(crossfade_seconds * sr)
    start_a_samples = int(switch_a * sr)
    end_a_samples = min(start_a_samples + crossfade_samples, stretched_a.shape[-1])
    body_a = stretched_a[..., :start_a_samples]
    tail_a = stretched_a[..., start_a_samples:end_a_samples]

    start_b_samples = int(switch_b * sr)
    head_b_end = min(start_b_samples + crossfade_samples, stretched_b.shape[-1])
    head_b = stretched_b[..., start_b_samples:head_b_end]
    body_b = stretched_b[..., head_b_end:]

    n = min(tail_a.shape[-1], head_b.shape[-1])
    if n == 0:
        raise ValueError(
            "Overlap region is empty — tracks too short or skip offsets too aggressive."
        )
    tail_a = tail_a[..., :n]
    head_b = head_b[..., :n]

    # Build the transition segment.
    if ai_bridge_seconds > 0:
        from songfuse.ai_bridge import generate_bridge

        bridge = ensure_stereo(
            generate_bridge(
                tail_a, head_b, sr,
                duration=ai_bridge_seconds,
                prompt=ai_bridge_prompt,
                seed=ai_bridge_seed,
            )
        )
        glue = int(min(0.5 * sr, tail_a.shape[-1] // 2, bridge.shape[-1] // 2, head_b.shape[-1] // 2))
        transition_audio = _sequence_with_bridge(tail_a, bridge, head_b, glue)
        notes.append(f"Inserted {ai_bridge_seconds:.2f}s MusicGen bridge.")
    elif style == "stems":
        from songfuse.stems import bass_swap_via_stems

        transition_audio = bass_swap_via_stems(tail_a, head_b, sr)
        notes.append("Used Demucs stem-based bass swap for the overlap.")
    else:
        transition_audio = crossfade(style, tail_a, head_b, sr)

    output = np.concatenate([p for p in (body_a, transition_audio, body_b) if p.size], axis=-1)

    peak = float(np.max(np.abs(output)) + 1e-9)
    if peak > 0.99:
        output = output * (0.99 / peak)

    plan = FusePlan(
        target_bpm=float(chosen_bpm),
        bars=int(effective_bars),
        style=style,
        tempo_variance_ok=variance_ok,
        semitone_shift_a=semitone_shift,
        switch_time_a=round(switch_a, 3),
        switch_time_b=round(switch_b, 3),
        crossfade_seconds=round(crossfade_seconds, 3),
        ai_bridge_seconds=float(ai_bridge_seconds),
        skip_head_a=float(skip_head_a),
        skip_tail_a=float(skip_tail_a),
        skip_head_b=float(skip_head_b),
        skip_tail_b=float(skip_tail_b),
        notes=notes,
    )

    result = FuseResult(audio=output, sr=sr, plan=plan,
                        analysis_a=a_analysis, analysis_b=b_analysis)

    if out is not None:
        save(out, output, sr)
    return result


def _rewrap_analysis(src: TrackAnalysis, downbeats: np.ndarray, duration: float,
                     stretch_rate: float) -> TrackAnalysis:
    """Return a lightweight TrackAnalysis with timings rescaled by `stretch_rate`.

    With librosa's convention (rate>1 shortens), new_time = old_time / rate and
    new_bpm = old_bpm * rate."""
    rate = max(stretch_rate, 1e-6)
    return TrackAnalysis(
        path=src.path,
        sr=src.sr,
        duration=duration,
        bpm=src.bpm * rate,
        beats=src.beats / rate,
        downbeats=downbeats,
        key_pc=src.key_pc,
        key_mode=src.key_mode,
        key_confidence=src.key_confidence,
        segments=[type(s)(start=s.start / rate, end=s.end / rate, kind=s.kind)
                  for s in src.segments],
    )


def _crossfade_tails(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Equal-power glue: last n samples of a fade into first n of b."""
    if n <= 0 or a.shape[-1] == 0 or b.shape[-1] == 0:
        return np.concatenate([a, b], axis=-1)
    n = min(n, a.shape[-1], b.shape[-1])
    tail_a = a[..., -n:]
    head_b = b[..., :n]
    body_a = a[..., :-n]
    body_b = b[..., n:]
    glue = constant_power_crossfade(tail_a, head_b)
    return np.concatenate([body_a, glue, body_b], axis=-1)


def _sequence_with_bridge(tail_a: np.ndarray, bridge: np.ndarray, head_b: np.ndarray,
                          glue: int) -> np.ndarray:
    """Compose tail_a → bridge → head_b with short equal-power crossfades at each seam."""
    a_plus_bridge = _crossfade_tails(tail_a, bridge, glue)
    return _crossfade_tails(a_plus_bridge, head_b, glue)


def batch_fuse(
    paths: list[str | Path],
    *,
    out: str | Path | None = None,
    sr: int = DEFAULT_SR,
    **fuse_kwargs,
) -> FuseResult:
    """Chain-fuse a list of tracks end-to-end (playlist-style auto-mix).

    Each intermediate mix is written to a temp file and re-fed to `fuse()` so the
    next track blends into the already-mixed tail."""
    import tempfile

    if len(paths) < 2:
        raise ValueError("batch_fuse requires at least 2 tracks.")

    fuse_kwargs.setdefault("sr", sr)
    result = fuse(paths[0], paths[1], **fuse_kwargs)
    for next_path in paths[2:]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            save(tmp_path, result.audio, result.sr)
            result = fuse(tmp_path, next_path, **fuse_kwargs)
        finally:
            tmp_path.unlink(missing_ok=True)

    if out is not None:
        save(out, result.audio, result.sr)
    return result
