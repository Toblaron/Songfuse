"""`songfuse` command-line interface."""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from songfuse import __version__, analyze, fuse
from songfuse.fuse import VALID_STYLES, batch_fuse

app = typer.Typer(
    add_completion=False,
    help="Fuse two songs with a seamless, beat-matched, harmonically aware transition.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


class Style(str, Enum):
    linear = "linear"
    equal_power = "equal_power"
    eq = "eq"
    bass_swap = "bass_swap"
    stems = "stems"


class TempoStrategy(str, Enum):
    a = "a"
    b = "b"
    avg = "avg"


class AiMode(str, Enum):
    bidirectional = "bidirectional"
    forward = "forward"
    backward = "backward"


class AiModel(str, Enum):
    small = "facebook/musicgen-small"
    medium = "facebook/musicgen-medium"
    large = "facebook/musicgen-large"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"songfuse {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Print version and exit.",
    ),
) -> None:
    """Songfuse: AI-assisted seamless audio transitions."""


def _default_out(a: Path, b: Path) -> Path:
    return Path(f"{a.stem}__x__{b.stem}.wav")


def _print_plan(summary: dict) -> None:
    plan = summary["plan"]
    table = Table(title="Transition plan")
    table.add_column("field"); table.add_column("value")
    for k, v in plan.items():
        if k == "notes":
            continue
        table.add_row(k, str(v))
    console.print(table)
    for note in plan.get("notes", []):
        console.print(f"[yellow]•[/yellow] {note}")


def _friendly_run(work):
    """Run a callable, converting known extras errors into clean CLI errors."""
    try:
        return work()
    except KeyboardInterrupt:
        err_console.print("[red]Interrupted.[/red]")
        raise typer.Exit(130)
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        if name in {"AIBridgeUnavailable", "StemsUnavailable"}:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2)
        raise


@app.command("analyze")
def analyze_cmd(
    paths: list[Path] = typer.Argument(..., exists=True, readable=True,
                                       help="One or more audio files to analyze."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Print BPM, key, Camelot code, and rough structural segments."""
    results = [analyze(p).to_dict() for p in paths]
    if json_out:
        console.print_json(data=results if len(results) > 1 else results[0])
        return
    for path, data in zip(paths, results):
        table = Table(title=f"Analysis: {path.name}")
        table.add_column("field"); table.add_column("value")
        for k in ("duration", "bpm", "key", "camelot", "key_confidence", "n_beats", "n_downbeats"):
            table.add_row(k, str(data[k]))
        console.print(table)


@app.command("fuse")
def fuse_cmd(
    track_a: Path = typer.Argument(..., exists=True, readable=True, help="Outgoing track."),
    track_b: Path = typer.Argument(..., exists=True, readable=True, help="Incoming track."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output WAV path (defaults to '<A>__x__<B>.wav' in CWD).",
    ),
    style: Style = typer.Option(Style.eq, "--style", "-s", help="Crossfade style."),
    bars: int = typer.Option(16, "--bars", "-b", min=1, max=64,
                             help="Crossfade length in bars (4 beats each)."),
    tempo: TempoStrategy = typer.Option(TempoStrategy.b, "--tempo",
                                        help="Target BPM strategy: a / b / avg."),
    target_bpm: Optional[float] = typer.Option(None, "--target-bpm",
                                               help="Override target BPM."),
    no_key_match: bool = typer.Option(False, "--no-key-match",
                                      help="Skip Camelot pitch shifting."),
    lufs: float = typer.Option(-14.0, "--lufs", help="Target integrated loudness (LUFS)."),
    skip_head_a: float = typer.Option(0.0, "--skip-head-a",
                                      help="Seconds to trim from the start of track A."),
    skip_tail_a: float = typer.Option(0.0, "--skip-tail-a",
                                      help="Seconds to trim from the end of track A."),
    skip_head_b: float = typer.Option(0.0, "--skip-head-b",
                                      help="Seconds to trim from the start of track B."),
    skip_tail_b: float = typer.Option(0.0, "--skip-tail-b",
                                      help="Seconds to trim from the end of track B."),
    ai_bridge_seconds: float = typer.Option(
        0.0, "--ai-bridge-seconds",
        help="If >0, insert a MusicGen-generated bridge of this length. Requires [ai] extras.",
    ),
    ai_mode: AiMode = typer.Option(
        AiMode.bidirectional, "--ai-mode",
        help="bidirectional = condition on A's outro AND B's intro (recommended). "
             "forward = A only; backward = B only.",
    ),
    ai_model: AiModel = typer.Option(
        AiModel.medium, "--ai-model",
        help="MusicGen checkpoint. 'large' sounds best but needs ~8GB VRAM.",
    ),
    ai_prompt: Optional[str] = typer.Option(None, "--ai-prompt",
                                            help="Optional text prompt for the AI bridge."),
    ai_seed: Optional[int] = typer.Option(None, "--ai-seed",
                                          help="Seed for reproducible AI bridge generation."),
    preview: bool = typer.Option(False, "--preview",
                                 help="Analyze + print the plan without rendering audio."),
    json_plan: bool = typer.Option(False, "--json-plan",
                                   help="Print the full transition plan as JSON."),
) -> None:
    """Render a seamless fused audio file."""
    if output is None and not preview:
        output = _default_out(track_a, track_b)

    if preview:
        a_info = analyze(track_a).to_dict()
        b_info = analyze(track_b).to_dict()
        table = Table(title="Preview — Track A vs Track B")
        table.add_column("field"); table.add_column("A"); table.add_column("B")
        for k in ("duration", "bpm", "key", "camelot", "key_confidence"):
            table.add_row(k, str(a_info[k]), str(b_info[k]))
        console.print(table)
        console.print(f"[dim]No audio rendered. Drop --preview to fuse to {_default_out(track_a, track_b)}.[/dim]")
        return

    def work():
        with console.status(f"[bold cyan]Fusing {track_a.name} + {track_b.name}..."):
            return fuse(
                track_a, track_b,
                out=output,
                style=style.value,
                bars=bars,
                match_key=not no_key_match,
                tempo_strategy=tempo.value,
                target_bpm=target_bpm,
                lufs_target=lufs,
                skip_head_a=skip_head_a,
                skip_tail_a=skip_tail_a,
                skip_head_b=skip_head_b,
                skip_tail_b=skip_tail_b,
                ai_bridge_seconds=ai_bridge_seconds,
                ai_bridge_prompt=ai_prompt,
                ai_bridge_seed=ai_seed,
                ai_bridge_mode=ai_mode.value,
                ai_bridge_model=ai_model.value,
            )

    result = _friendly_run(work)
    summary = result.to_summary()
    if json_plan:
        console.print_json(data=summary)
        return

    console.print(f"[green]Wrote[/green] [bold]{output}[/bold]  "
                  f"({summary['output_seconds']}s @ {result.sr}Hz)")
    _print_plan(summary)


@app.command("mix")
def mix_cmd(
    playlist: Path = typer.Argument(
        ..., exists=True, readable=True,
        help="Playlist file (one audio path per line) OR a directory of audio files.",
    ),
    output: Path = typer.Option(Path("mix.wav"), "--output", "-o", help="Output WAV path."),
    style: Style = typer.Option(Style.eq, "--style", "-s"),
    bars: int = typer.Option(16, "--bars", "-b", min=1, max=64),
    tempo: TempoStrategy = typer.Option(TempoStrategy.b, "--tempo"),
    lufs: float = typer.Option(-14.0, "--lufs"),
    no_key_match: bool = typer.Option(False, "--no-key-match"),
) -> None:
    """Chain-fuse an entire playlist end-to-end (continuous DJ-style mix)."""
    if playlist.is_dir():
        exts = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
        paths = sorted(p for p in playlist.iterdir() if p.suffix.lower() in exts)
    else:
        paths = [Path(line).expanduser()
                 for line in playlist.read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    if len(paths) < 2:
        err_console.print("[red]Need at least 2 tracks.[/red]")
        raise typer.Exit(2)
    for p in paths:
        if not p.exists():
            err_console.print(f"[red]Missing track: {p}[/red]")
            raise typer.Exit(2)

    console.print(f"[bold]Mixing {len(paths)} tracks[/bold] → {output}")
    with console.status("Fusing playlist..."):
        result = _friendly_run(lambda: batch_fuse(
            paths, out=output, style=style.value, bars=bars,
            tempo_strategy=tempo.value, lufs_target=lufs, match_key=not no_key_match,
        ))
    console.print(f"[green]Wrote[/green] {output} ({result.audio.shape[-1] / result.sr:.1f}s)")


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Run the FastAPI HTTP service."""
    import uvicorn

    uvicorn.run("songfuse.api:app", host=host, port=port, reload=reload)


@app.command("styles")
def styles_cmd() -> None:
    """List available crossfade styles."""
    table = Table(title="Crossfade styles")
    table.add_column("name"); table.add_column("summary")
    docs = {
        "linear": "Naive linear fade — expect a midpoint dip.",
        "equal_power": "Constant-power sin/cos — flat RMS across the overlap.",
        "eq": "Bass-swap via Butterworth crossover at 180Hz + equal-power highs.",
        "bass_swap": "Alias for 'eq'.",
        "stems": "Demucs-based stem swap: B's bass+drums + A's vocals/other. Requires [stems] extras.",
    }
    for name in VALID_STYLES:
        table.add_row(name, docs[name])
    console.print(table)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
