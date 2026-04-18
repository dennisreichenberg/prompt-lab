"""CLI entry point for prompt-lab."""

from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .ollama import OllamaClient
from .templates import all_templates, delete_template, get_template, save_template

app = typer.Typer(
    name="prompt-lab",
    help="Side-by-side prompt testing across local Ollama models.",
    add_completion=False,
)
console = Console()
err = Console(stderr=True)


def _client(host: str) -> OllamaClient:
    return OllamaClient(base_url=host)


def _ollama_error(exc: Exception) -> None:
    err.print(f"\n[red]Cannot reach Ollama:[/red] {exc}")
    err.print("[dim]Is Ollama running? Try: ollama serve[/dim]")
    raise typer.Exit(1)


def _resolve_models(models_str: str, host: str) -> list[str]:
    if models_str.strip().lower() == "all":
        try:
            names = _client(host).list_model_names()
        except Exception as exc:
            _ollama_error(exc)
        if not names:
            err.print("[yellow]No models installed.[/yellow]")
            raise typer.Exit(1)
        return names
    return [m.strip() for m in models_str.split(",") if m.strip()]


@app.command("run")
def run_prompt(
    prompt: Optional[str] = typer.Argument(None, help="Prompt text to send."),
    models: str = typer.Option(
        ..., "--models", "-m", help="Comma-separated model names, or 'all'."
    ),
    file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="Read prompt from a file instead."
    ),
    template: Optional[str] = typer.Option(
        None, "--template", "-t", help="Use a saved named template."
    ),
    host: str = typer.Option("http://localhost:11434", "--host", "-H", help="Ollama base URL."),
) -> None:
    """Run a prompt against multiple models side-by-side."""
    if template:
        text = get_template(template)
        if text is None:
            err.print(f"[red]Template not found:[/red] {template}")
            raise typer.Exit(1)
    elif file:
        try:
            text = file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            err.print(f"[red]Cannot read file:[/red] {exc}")
            raise typer.Exit(1)
    elif prompt:
        text = prompt
    else:
        err.print("[red]Provide a prompt, --file, or --template.[/red]")
        raise typer.Exit(1)

    model_list = _resolve_models(models, host)

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]prompt-lab[/bold cyan] v{__version__} — Running on {len(model_list)} model(s)",
            border_style="cyan",
        )
    )
    console.print()
    console.print(Panel(text, title="[bold]Prompt[/bold]", border_style="dim", expand=False))
    console.print()
    console.print(f"[dim]Querying {len(model_list)} model(s) in parallel…[/dim]\n")

    client = _client(host)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(model_list)) as pool:
        futures = {pool.submit(client.generate, m, text): m for m in model_list}
        results = []
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: r.elapsed_seconds)

    panels = []
    for r in results:
        if r.error:
            body = f"[red]Error:[/red] {r.error}"
            border = "red"
        else:
            body = r.response
            border = "green"
        title = f"[bold]{r.model}[/bold]  [dim]{r.elapsed_seconds:.2f}s[/dim]"
        panels.append(Panel(body, title=title, border_style=border, expand=True))

    console.print(Columns(panels, equal=True, expand=True))
    console.print()

    timing_table = Table(title="Timing summary", box=box.SIMPLE, border_style="dim")
    timing_table.add_column("Model", style="cyan")
    timing_table.add_column("Time", style="green", justify="right")
    timing_table.add_column("Status", style="white")
    for r in results:
        status = "[red]error[/red]" if r.error else "[green]ok[/green]"
        timing_table.add_row(r.model, f"{r.elapsed_seconds:.2f}s", status)
    console.print(timing_table)
    console.print()


@app.command("save")
def save_cmd(
    name: str = typer.Argument(..., help="Template name."),
    prompt: str = typer.Argument(..., help="Prompt text to save."),
) -> None:
    """Save a named prompt template for later reuse."""
    save_template(name, prompt)
    console.print(f"\n[green]✓[/green] Saved template [bold cyan]{name}[/bold cyan].\n")


@app.command("list")
def list_cmd() -> None:
    """List all saved prompt templates."""
    templates = all_templates()
    if not templates:
        console.print("\n[dim]No templates saved yet. Use: prompt-lab save <name> \"<prompt>\"[/dim]\n")
        return

    console.print()
    table = Table(title="Saved Templates", box=box.ROUNDED, border_style="dim")
    table.add_column("Name", style="bold cyan", no_wrap=True)
    table.add_column("Preview", style="white")

    for name, text in sorted(templates.items()):
        preview = text.replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "…"
        table.add_row(name, preview)

    console.print(table)
    console.print(f"\n[dim]{len(templates)} template(s) stored.[/dim]\n")


@app.command("delete")
def delete_cmd(
    name: str = typer.Argument(..., help="Template name to delete."),
) -> None:
    """Delete a saved prompt template."""
    if delete_template(name):
        console.print(f"\n[green]✓[/green] Deleted template [bold]{name}[/bold].\n")
    else:
        err.print(f"\n[red]Template not found:[/red] {name}\n")
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
