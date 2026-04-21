"""FastAPI service: JSON API + a drag-drop web UI."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from songfuse import __version__, analyze, fuse

_WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(
    title="Songfuse",
    version=__version__,
    description="Generate a seamless transition between two audio tracks.",
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_WEB_DIR / "index.html", media_type="text/html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


def _save_upload(upload: UploadFile, tmpdir: Path) -> Path:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    suffix = Path(upload.filename).suffix or ".bin"
    target = tmpdir / f"{Path(upload.filename).stem}{suffix}"
    with target.open("wb") as f:
        f.write(upload.file.read())
    return target


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    with tempfile.TemporaryDirectory() as tmp:
        path = _save_upload(file, Path(tmp))
        return JSONResponse(analyze(path).to_dict())


@app.post("/fuse", response_model=None)
async def fuse_endpoint(
    track_a: UploadFile = File(..., description="Outgoing track."),
    track_b: UploadFile = File(..., description="Incoming track."),
    style: str = Form("eq"),
    bars: int = Form(16),
    tempo_strategy: str = Form("b"),
    target_bpm: float | None = Form(None),
    match_key: bool = Form(True),
    lufs: float = Form(-14.0),
    skip_head_a: float = Form(0.0),
    skip_tail_a: float = Form(0.0),
    skip_head_b: float = Form(0.0),
    skip_tail_b: float = Form(0.0),
    ai_bridge_seconds: float = Form(0.0),
    ai_mode: str = Form("bidirectional"),
    ai_model: str = Form("facebook/musicgen-medium"),
    ai_prompt: str | None = Form(None),
    ai_seed: int | None = Form(None),
    return_json: bool = Form(False),
) -> StreamingResponse | JSONResponse:
    """Upload two audio files; receive a fused WAV (or the transition plan as JSON)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        path_a = _save_upload(track_a, tmp_path)
        path_b = _save_upload(track_b, tmp_path)
        try:
            result = fuse(
                path_a, path_b,
                style=style,
                bars=bars,
                match_key=match_key,
                tempo_strategy=tempo_strategy,
                target_bpm=target_bpm,
                lufs_target=lufs,
                skip_head_a=skip_head_a,
                skip_tail_a=skip_tail_a,
                skip_head_b=skip_head_b,
                skip_tail_b=skip_tail_b,
                ai_bridge_seconds=ai_bridge_seconds,
                ai_bridge_prompt=ai_prompt,
                ai_bridge_seed=ai_seed,
                ai_bridge_mode=ai_mode,
                ai_bridge_model=ai_model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if return_json:
        return JSONResponse(result.to_summary())

    buffer = io.BytesIO()
    audio = result.audio
    if audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
        data = audio.T
    else:
        data = audio
    sf.write(buffer, data.astype(np.float32, copy=False), result.sr,
             subtype="PCM_16", format="WAV")
    buffer.seek(0)
    headers = {"X-Songfuse-Plan": _safe_header(result.to_summary())}
    return StreamingResponse(buffer, media_type="audio/wav", headers=headers)


def _safe_header(summary: dict) -> str:
    import json

    try:
        return json.dumps(summary, separators=(",", ":"))[:4000]
    except Exception:  # noqa: BLE001
        return ""
