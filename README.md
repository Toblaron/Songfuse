# Songfuse

Fuse two songs into one with an AI-assisted, beat-matched, harmonically aware
transition — inspired by the architecture surveyed in
*Automated Audio Fusion: DSP + Generative AI* (Gemini, 2026).

Songfuse analyzes both tracks (tempo, key, downbeats, structural segments),
normalizes them to a common loudness target (EBU R128), pitch-shifts for
Camelot-wheel harmonic compatibility, tempo-matches via time-stretching, and
renders the overlap with a constant-power or bass-swap EQ crossfade — the same
moves a professional DJ would make by hand. Optional extras plug in
[Demucs](https://github.com/facebookresearch/demucs) for stem-level bass swaps
and [MusicGen](https://github.com/facebookresearch/audiocraft) for a generative
bridge between tracks whose tempos or genres are too far apart to crossfade.

## Install

```bash
pip install -e .                 # core DSP + CLI + API
pip install -e ".[stems]"        # adds Demucs-based stem separation
pip install -e ".[ai]"           # adds MusicGen generative bridge
pip install -e ".[dev]"          # + pytest
```

Python ≥ 3.10 is required. `librosa` pulls in `numba`; first import will
take a moment while it JIT-compiles.

## CLI

```bash
# Analyze a single track: BPM, key, Camelot code, structure.
songfuse analyze track.mp3

# Render a fused WAV. Default output is "<A>__x__<B>.wav" in the CWD.
songfuse fuse song_a.mp3 song_b.mp3

# Pick a crossfade style: linear | equal_power | eq | bass_swap | stems
songfuse fuse a.mp3 b.mp3 --style stems        # Demucs-powered bass swap (requires [stems])
songfuse fuse a.mp3 b.mp3 --style equal_power --bars 8

# Trim intros/outros that shouldn't be mixed.
songfuse fuse a.mp3 b.mp3 --skip-tail-a 5 --skip-head-b 20

# Plan-only dry run — no audio rendered.
songfuse fuse a.mp3 b.mp3 --preview

# Force a target BPM and skip key matching.
songfuse fuse a.mp3 b.mp3 --target-bpm 124 --no-key-match

# Splice a 4-second MusicGen bridge between A and B (requires [ai] extras).
songfuse fuse a.mp3 b.mp3 --ai-bridge-seconds 4 --ai-seed 42 \
    --ai-prompt "deep house to drum and bass, 16 bar breakdown with risers"

# Chain-fuse an entire playlist or folder into one continuous mix.
songfuse mix tracks/ -o set.wav --bars 16
songfuse mix playlist.txt -o set.wav   # one path per line; '#' comments allowed

# Discover available styles.
songfuse styles
```

## Web UI + HTTP API

```bash
songfuse serve --port 8000
```

Open **http://localhost:8000** for the drag-drop web UI: drop two audio files,
pick a style / bars / tempo target, click *Fuse*, play or download the result.
Both tracks get analyzed on upload so you see BPM, key, and Camelot code before
committing to the transition.

Endpoints:

- `GET /` — the web UI (static HTML + JS, no build step).

- `POST /analyze` — multipart upload `file` → JSON with BPM / key / segments.
- `POST /fuse` — multipart upload `track_a`, `track_b`, plus optional form
  fields (`style`, `bars`, `tempo_strategy`, `target_bpm`, `match_key`,
  `lufs`, `ai_bridge_seconds`, `ai_prompt`, `return_json`).
  Returns a WAV stream; set `return_json=true` to receive the transition plan
  instead.

Example:

```bash
curl -X POST http://localhost:8000/fuse \
  -F track_a=@a.mp3 -F track_b=@b.mp3 \
  -F style=eq -F bars=16 -F lufs=-14 \
  --output fused.wav
```

## Python

```python
from songfuse import analyze, fuse

info = analyze("song.mp3")
print(info.bpm, info.key_name, info.camelot)

result = fuse("a.mp3", "b.mp3",
              out="fused.wav",
              style="eq",              # linear | equal_power | eq | bass_swap
              bars=16,
              match_key=True,
              tempo_strategy="b",      # "a" | "b" | "avg"
              lufs_target=-14.0,
              ai_bridge_seconds=0.0)   # set >0 to insert MusicGen bridge
print(result.plan)
```

## Architecture

The package maps one-to-one onto the reference's three-pipeline design:

| Pipeline        | Module                 | Responsibility                                        |
|-----------------|------------------------|-------------------------------------------------------|
| **Annotation**  | `songfuse.analysis`    | BPM / beats / downbeats, Krumhansl key, segments.     |
|                 | `songfuse.stems`       | Optional Demucs stem separation.                      |
| **Logic**       | `songfuse.camelot`     | Camelot-wheel compatibility + minimal-shift solver.   |
|                 | `songfuse.alignment`   | Tempo matching, pitch shifting, downbeat alignment.   |
| **Generation**  | `songfuse.loudness`    | EBU R128 LUFS normalization.                          |
|                 | `songfuse.transition`  | Constant-power and bass-swap crossfades.              |
|                 | `songfuse.ai_bridge`   | Optional MusicGen bridge (audio inpainting).          |
| **Interfaces**  | `songfuse.fuse`        | Orchestrator returning a `FuseResult`.                |
|                 | `songfuse.cli`         | Typer CLI (`songfuse analyze|fuse|serve`).            |
|                 | `songfuse.api`         | FastAPI service (`/analyze`, `/fuse`, `/health`).     |

Defaults lean on librosa (MIR + time-stretch / pitch-shift), scipy (Butterworth
crossovers), pyloudnorm (K-weighted loudness), Typer (CLI), and FastAPI (HTTP).
Swap in madmom / Essentia / EDMFormer for production-grade annotation, and
Discrete Diffusion Models (DAC / Encodec tokens) for higher-fidelity
bidirectional inpainting than the bundled MusicGen continuation path.

## Tests

```bash
pytest
```

Pure-DSP tests are deterministic and offline. The end-to-end test builds
synthetic tone-plus-click files and renders a fused WAV — it will be skipped
automatically if `librosa` is not installed.

## Limitations & future work

- Structural segmentation is energy-based; wire up MSAF / EDMFormer for
  genre-aware switch points.
- Downbeat tracking is every-fourth-beat heuristic; madmom's
  `DBNBeatTrackingProcessor` is a drop-in upgrade.
- MusicGen bridge uses continuation; swap to a discrete-diffusion inpainting
  model for bidirectional conditioning on both A's outro and B's intro.
- True-peak limiter is a simple peak normalizer; replace with an ISP limiter
  (e.g. `pyloudnorm`'s true-peak meter + soft clipper) for broadcast-safe output.
