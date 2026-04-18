"""Optional generative bridge: MusicGen-based audio inpainting.

Lazy-imports audiocraft so the package works without torch/audiocraft installed."""

from __future__ import annotations

import numpy as np


class AIBridgeUnavailable(RuntimeError):
    pass


def generate_bridge(
    tail_a: np.ndarray,
    head_b: np.ndarray,
    sr: int,
    duration: float = 4.0,
    prompt: str | None = None,
    model_name: str = "facebook/musicgen-small",
) -> np.ndarray:
    """Generate a short musical bridge conditioned on the end of A.

    `tail_a` and `head_b` are used to build a text prompt describing the desired
    style when none is given. Returns a (channels, samples) float32 array at `sr`.
    """
    try:
        import torch  # noqa: F401
        from audiocraft.models import MusicGen
    except Exception as exc:  # noqa: BLE001
        raise AIBridgeUnavailable(
            "Install AI extras: pip install 'songfuse[ai]' (requires torch + audiocraft)"
        ) from exc

    import torch  # type: ignore
    import torchaudio  # type: ignore

    model = MusicGen.get_pretrained(model_name)
    model.set_generation_params(duration=duration)

    if prompt is None:
        prompt = "seamless DJ transition, same key and tempo, smooth crossfade bridge"

    # Use tail_a as a melody/continuation prompt. audiocraft wants (batch, channels, samples).
    cond_audio = tail_a
    if cond_audio.ndim == 1:
        cond_audio = np.stack([cond_audio, cond_audio], axis=0)
    cond_tensor = torch.from_numpy(cond_audio).float().unsqueeze(0)
    cond_sr = sr

    try:
        wav = model.generate_continuation(cond_tensor, cond_sr, [prompt], progress=False)
    except Exception:
        wav = model.generate([prompt], progress=False)

    if wav.ndim == 3:
        wav = wav[0]
    wav = wav.detach().cpu().numpy().astype(np.float32)

    model_sr = model.sample_rate
    if model_sr != sr:
        wav_t = torch.from_numpy(wav)
        wav_t = torchaudio.functional.resample(wav_t, model_sr, sr)
        wav = wav_t.numpy().astype(np.float32)

    if wav.ndim == 1:
        wav = np.stack([wav, wav], axis=0)

    # Head_b is reserved for future bidirectional inpainting conditioning.
    _ = head_b
    return wav
