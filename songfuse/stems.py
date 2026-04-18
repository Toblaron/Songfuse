"""Optional Demucs-powered stem separation.

Pip-installable via `pip install songfuse[stems]`. We lazy-import so the core
package remains lightweight. Output shape: dict of (channels, samples) float32."""

from __future__ import annotations

import numpy as np


STEM_NAMES = ("drums", "bass", "other", "vocals")


class StemsUnavailable(RuntimeError):
    pass


def separate(audio: np.ndarray, sr: int, model: str = "htdemucs") -> dict[str, np.ndarray]:
    """Separate `audio` (channels, samples) into Demucs stems.

    Raises StemsUnavailable if `demucs` / `torch` are not installed."""
    try:
        import torch  # noqa: F401
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except Exception as exc:  # noqa: BLE001
        raise StemsUnavailable(
            "Install stem-separation extras: pip install 'songfuse[stems]'"
        ) from exc

    import torch  # type: ignore

    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] == 1:
        audio = np.concatenate([audio, audio], axis=0)

    loaded = get_model(model)
    loaded.cpu().eval()
    tensor = torch.from_numpy(audio).unsqueeze(0).float()
    with torch.no_grad():
        sources = apply_model(loaded, tensor, device="cpu", split=True, overlap=0.1)[0]
    sources = sources.cpu().numpy()

    return {name: sources[i].astype(np.float32, copy=False)
            for i, name in enumerate(loaded.sources)}


def bass_swap_via_stems(a: np.ndarray, b: np.ndarray, sr: int) -> np.ndarray:
    """Compose a transition by taking the bass stem from B and melody/vocal from A.

    Requires demucs. Intended as a richer alternative to filter-based bass swapping."""
    stems_a = separate(a, sr)
    stems_b = separate(b, sr)
    n = min(min(v.shape[-1] for v in stems_a.values()),
            min(v.shape[-1] for v in stems_b.values()))

    def take(stems, key):
        return stems[key][..., :n] if key in stems else np.zeros_like(stems["drums"][..., :n])

    mix = (
        take(stems_b, "bass")
        + take(stems_b, "drums") * 0.8
        + take(stems_a, "vocals") * 0.7
        + take(stems_a, "other") * 0.6
    )
    peak = float(np.max(np.abs(mix)) + 1e-9)
    if peak > 0.99:
        mix = mix * (0.99 / peak)
    return mix.astype(np.float32, copy=False)
