"""`songfuse` command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from songfuse import analyze, fuse

app = typer.Typer(
    add_completion=False,
    help="Fuse two songs with a seamless, beat-matched, harmonically aware transition.",
    no_args_is_help=True,
)
console = Console()


@app.command("analyze")
def analyze_cmd(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Audio file to analyze."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Print BPM, key, Camelot code, and rough structural segments for a single track."""
    data = analyze(path).to_dict()
    if json_out:
        console.print_json(data=data)
        return
    table = Table(title=f"Analysis: {path.name}")
    table.add_column("field")
    table.add_column("value")
    for k in ("duration", "bpm", "key", "camelot", "key_confidence", "n_beats", "n_downbeats"):
        table.add_row(k, str(data[k]))
    console.print(table)
    if data["segments"]:
        seg_table = Table(title="Structural segments")
        seg_table.add_column("start"); seg_table.add_column("end"); seg_table.add_column("kind")
        for s in data["segments"]:
            seg_table.add_row(str(s["start"]), str(s["end"]), s["kind"])
        console.print(seg_table)


@app.command("fuse")
def fuse_cmd(
    track_a: Path = typer.Argument(..., exists=True, readable=True, help="Outgoing track."),
    track_b: Path = typer.Argument(..., exists=True, readable=True, help="Incoming track."),
    output: Path = typer.Option(Path("fused.wav"), "--output", "-o", help="Output WAV path."),
    style: str = typer.Option("eq", "--style", "-s",
                              help="Crossfade style: linear | equal_power | eq | bass_swap."),
    bars: int = typer.Option(16, "--bars", "-b", help="Crossfade length in bars (4 beats each)."),
    tempo_strategy: str = typer.Option("b", "--tempo", help="Target BPM: a | b | avg."),
    target_bpm: float = typer.Option(None, "--target-bpm", help="Override target BPM."),
    no_key_match: bool = typer.Option(False, "--no-key-match", help="Skip Camelot pitch shifting."),
    lufs: float = typer.Option(-14.0, "--lufs", help="Target integrated loudness (LUFS)."),
    ai_bridge_seconds: float = typer.Option(
        0.0, "--ai-bridge-seconds",
        help="If >0, insert a MusicGen-generated bridge of this length. Requires [ai] extras.",
    ),
    ai_prompt: str = typer.Option(None, "--ai-prompt", help="Optional text prompt for the AI bridge."),
    json_plan: bool = typer.Option(False, "--json-plan", help="Print the transition plan as JSON."),
) -> None:
    """Render a seamless fused audio file."""
    with console.status(f"[bold cyan]Fusing {track_a.name} + {track_b.name}..."):
        result = fuse(
            track_a, track_b,
            out=output,
            style=style,
            bars=bars,
            match_key=not no_key_match,
            tempo_strategy=tempo_strategy,
            target_bpm=target_bpm,
            lufs_target=lufs,
            ai_bridge_seconds=ai_bridge_seconds,
            ai_bridge_prompt=ai_prompt,
        )
    summary = result.to_summary()
    if json_plan:
        console.print_json(data=summary)
        return

    console.print(f"[green]Wrote[/green] [bold]{output}[/bold]  "
                  f"({summary['output_seconds']}s @ {result.sr}Hz)")
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


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Run the FastAPI HTTP service."""
    import uvicorn

    uvicorn.run("songfuse.api:app", host=host, port=port, reload=reload)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
