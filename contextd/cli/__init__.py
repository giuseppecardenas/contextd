"""Contextd CLI — package root.

Exports the Click group ``cli`` and the entry point ``main``. Commands
are registered by importing the sub-modules below; each sub-module uses
the ``cli`` symbol defined here and attaches commands via
``@cli.command(...)``.

Split layout (SD #80):

- ``cli/__init__.py`` — group + ``init`` command + ``main`` entry
- ``cli/infra.py``    — ``up``, ``down``, ``status``
- ``cli/corpora.py``  — ``add-corpus``, ``list-corpora``, ``index``
- ``cli/query.py``    — ``ask``, ``logs``, ``costs``
- ``cli/_shared.py``  — ``PipelineDeps``, ``_load_cfg``, ``console`` (private helpers)

All commands route through the global config (``~/.contextd/config.toml``).
The factory layer stands up the Neo4j backend based on ``[storage] backend``.
"""

from __future__ import annotations

import os
import shutil
from importlib import resources

import click

from contextd._paths import contextd_home
from contextd.cli._shared import console


@click.group()
def cli() -> None:
    """Contextd — local GraphRAG knowledge layer."""


@cli.command()
@click.option("--yes", is_flag=True, help="Accept all defaults non-interactively.")
def init(yes: bool) -> None:
    """First-run wizard — creates ~/.contextd/ layout and registers MCP."""
    home = contextd_home()
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

    copied = 0
    for name in ("summarise.md", "relate.md", "translate.md"):
        dst = home / "prompts" / name
        if not dst.exists():
            src_text = (
                resources.files("contextd.prompts").joinpath(name).read_text(encoding="utf-8")
            )
            dst.write_text(src_text, encoding="utf-8")
            copied += 1
    if copied:
        console.print(f"[green]✓[/] prompt templates copied ({copied}/3, overridable)")
    else:
        console.print("[dim]·[/] prompt templates already present")

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

    # Docker check — all supported backends run via docker compose.
    if not shutil.which("docker"):
        console.print(
            "[yellow]⚠[/] docker not on PATH; install Docker before running `contextd up`"
        )

    console.print("\n[bold]next:[/] `contextd up` to start the daemon.")


def main() -> None:
    cli()


# Trigger command registration by importing the sub-modules. These
# imports MUST land after ``cli`` is defined so the decorators can
# attach to it.
from contextd.cli import corpora, infra, query  # noqa: E402,F401,I001


if __name__ == "__main__":
    main()
