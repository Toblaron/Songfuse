"""AI-composed transitions via MusicGen.

Instead of DJ-style crossfading real audio, this module generates a musical
bridge with a generative model. The key technique is **bidirectional
inpainting via reversal**:

1. Run MusicGen continuation forward from A's tail → produces audio that flows
   naturally out of A for `duration/2` seconds.
2. Reverse B's head in time, run MusicGen continuation on that → produces audio
   that flows into (reversed) B. Reverse the result back in time → you now have
   audio that flows naturally *into* B.
3. Crossfade the two generated halves in the middle with an equal-power curve.

The bridge is AI-composed content, not manipulated source audio. It is
musically coherent with A at the start and with B at the end because both
halves were conditioned on the real tracks."""

from __future__ import annotations

import numpy as np


DEFAULT_MODEL = "facebook/musicgen-medium"
VALID_MODES = ("bidirectional", "forward", "backward")


class AIBridgeUnavailable(RuntimeError):
    pass


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio[np.newaxis, :]
    if audio.shape[0] > 1:
        return audio.mean(axis=0, keepdims=True)
    return audio


def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=0)
    if audio.shape[0] == 1:
        return np.concatenate([audio, audio], axis=0)
    return audio


def generate_transition(
    tail_a: np.ndarray,
    head_b: np.ndarray,
    sr: int,
    duration: float = 8.0,
    mode: str = "bidirectional",
    model_name: str = DEFAULT_MODEL,
    prompt: str | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Generate an AI-composed transition between two tracks.

    Args:
        tail_a: last N seconds of the outgoing track, shape (channels, samples).
        head_b: first N seconds of the incoming track, shape (channels, samples).
        sr: sample rate of both inputs.
        duration: desired bridge length in seconds. For 'bidirectional' the two
            halves are each `duration/2` long.
        mode: 'bidirectional' (recommended), 'forward' (condition on A only),
            'backward' (condition on B only, reverse-generated).
        model_name: HuggingFace MusicGen model. 'facebook/musicgen-small',
            '-medium', or '-large'. Large needs ~8GB VRAM.
        prompt: text description to steer generation. If None, a neutral
            'smooth continuation' prompt is used.
        seed: torch RNG seed for reproducible generation.
        device: 'cuda', 'cpu', or None to auto-pick.

    Returns:
        (2, samples) float32 bridge at `sr`.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode {mode!r}. Valid: {VALID_MODES}")
    if duration <= 0:
        raise ValueError("duration must be positive.")

    try:
        import torch  # noqa: F401
        from audiocraft.models import MusicGen
    except Exception as exc:  # noqa: BLE001
        raise AIBridgeUnavailable(
            "Install AI extras: pip install 'songfuse[ai]' (needs torch + audiocraft)"
        ) from exc

    import torch  # type: ignore
    import torchaudio  # type: ignore

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    model = MusicGen.get_pretrained(model_name, device=device)
    model_sr = int(model.sample_rate)

    prompt = prompt or "smooth musical transition continuing the style, same tempo and key"

    # MusicGen-small/medium expect mono at model's sample rate. Resample + mono-fold.
    def _prep_cond(audio: np.ndarray) -> torch.Tensor:
        mono = _ensure_mono(np.asarray(audio, dtype=np.float32))  # (1, T)
        t = torch.from_numpy(mono).float()
        if sr != model_sr:
            t = torchaudio.functional.resample(t, sr, model_sr)
        return t.unsqueeze(0).to(device)  # (1, 1, T)

    def _to_np(t: torch.Tensor) -> np.ndarray:
        """Pull MusicGen output back to numpy at our working `sr`."""
        if t.ndim == 3:
            t = t[0]
        t = t.detach().cpu().float()
        if model_sr != sr:
            t = torchaudio.functional.resample(t, model_sr, sr)
        return t.numpy().astype(np.float32)

    def _continue(cond_audio: np.ndarray, length: float) -> np.ndarray:
        """Generate `length` seconds of continuation from `cond_audio`.

        audiocraft's generate_continuation returns prompt+generation concatenated,
        so we slice the tail to get just the generated portion."""
        prep = _prep_cond(cond_audio)
        cond_seconds = prep.shape[-1] / model_sr
        model.set_generation_params(duration=cond_seconds + length)
        with torch.no_grad():
            wav = model.generate_continuation(
                prep, prompt_sample_rate=model_sr,
                descriptions=[prompt], progress=False,
            )
        out = _to_np(wav)
        target_samples = int(length * sr)
        # Take the last `length` seconds — that's the generated continuation.
        if out.shape[-1] > target_samples:
            out = out[..., -target_samples:]
        return out

    if mode == "forward":
        return _ensure_stereo(_continue(tail_a, duration))

    if mode == "backward":
        reversed_b = np.ascontiguousarray(head_b[..., ::-1])
        gen = _continue(reversed_b, duration)
        return _ensure_stereo(np.ascontiguousarray(gen[..., ::-1]))

    # Bidirectional: two halves that meet in the middle.
    half = duration / 2.0
    forward = _continue(tail_a, half)
    reversed_b = np.ascontiguousarray(head_b[..., ::-1])
    backward_gen = _continue(reversed_b, half)
    backward = np.ascontiguousarray(backward_gen[..., ::-1])

    # Equal-power crossfade the two halves in the middle.
    from songfuse.transition import constant_power_crossfade

    n = min(forward.shape[-1], backward.shape[-1])
    bridge = constant_power_crossfade(forward[..., -n:], backward[..., :n])
    return _ensure_stereo(bridge)


# Back-compat alias for the old forward-only API.
def generate_bridge(
    tail_a: np.ndarray,
    head_b: np.ndarray,
    sr: int,
    duration: float = 4.0,
    prompt: str | None = None,
    model_name: str = DEFAULT_MODEL,
    seed: int | None = None,
) -> np.ndarray:
    """Legacy forward-only bridge (deprecated — use generate_transition instead)."""
    return generate_transition(
        tail_a, head_b, sr,
        duration=duration, mode="forward",
        model_name=model_name, prompt=prompt, seed=seed,
    )
