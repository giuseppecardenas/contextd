"""Contextd CLI.

All commands route through the global config (~/.contextd/config.toml).
The factory layer decides whether to stand up Memgraph or Kuzu based on
[storage] backend.
"""

from __future__ import annotations

import os
import shutil
from importlib import resources
from pathlib import Path

import click
from rich.console import Console

CONTEXTD_HOME = Path(os.environ.get("CONTEXTD_HOME", str(Path.home() / ".contextd")))
console = Console()


@click.group()
def cli() -> None:
    """Contextd — local GraphRAG knowledge layer."""


@cli.command()
@click.option("--yes", is_flag=True, help="Accept all defaults non-interactively.")
def init(yes: bool) -> None:
    """First-run wizard — creates ~/.contextd/ layout and registers MCP."""
    home = CONTEXTD_HOME
    for sub in ("corpora", "state", "state/session-log", "state/checkpoints", "logs", "prompts"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✓[/] created {home} layout")

    config_path = home / "config.toml"
    if not config_path.exists():
        default = resources.files("contextd").joinpath("default_config.toml").read_text()
        config_path.write_text(default)
        console.print(f"[green]✓[/] wrote default config to {config_path}")

    compose_path = home / "docker-compose.yml"
    if not compose_path.exists():
        compose = resources.files("contextd").joinpath("docker_compose.yml").read_text()
        compose_path.write_text(compose)
        console.print(f"[green]✓[/] wrote docker-compose template to {compose_path}")

    for name in ("summarise.md", "relate.md", "translate.md"):
        dst = home / "prompts" / name
        if not dst.exists():
            src_text = resources.files("prompts").joinpath(name).read_text()
            dst.write_text(src_text)
    console.print("[green]✓[/] prompt templates copied (overridable)")

    # Env-var prerequisite check.
    missing: list[str] = []
    if not os.environ.get("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY (get one at https://aistudio.google.com/app/apikey)")
    if not os.environ.get("VOYAGE_API_KEY"):
        missing.append("VOYAGE_API_KEY (get one at https://www.voyageai.com/)")
    if missing:
        console.print("[yellow]⚠[/] missing env vars — set these before `contextd up`:")
        for m in missing:
            console.print(f"  - {m}")

    # Docker check (informational only — KuzuDB backend doesn't need it).
    if not shutil.which("docker"):
        console.print(
            "[yellow]⚠[/] docker not on PATH; set [storage] backend = 'kuzu' to run without Docker"
        )

    console.print("\n[bold]next:[/] `contextd up` to start the daemon.")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
