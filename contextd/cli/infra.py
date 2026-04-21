"""Infra-management commands: ``up`` / ``down`` / ``status``."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import click

from contextd._paths import contextd_home
from contextd.cli import cli
from contextd.cli._shared import _load_cfg, console


@cli.command()
def up() -> None:
    """Start the storage backend and the indexer daemon."""
    cfg = _load_cfg()

    if cfg.storage.backend == "memgraph":
        if not shutil.which("docker"):
            raise click.ClickException(
                "docker not on PATH. Install Docker, or set [storage] backend = 'kuzu' "
                "in ~/.contextd/config.toml to run without it."
            )
        compose_file = Path(cfg.storage.memgraph.docker_compose_file).expanduser()
        subprocess.run(["docker", "compose", "-f", str(compose_file), "up", "-d"], check=True)
        console.print("[green]✓[/] memgraph container up at 127.0.0.1:7687")
    else:
        db_path = Path(cfg.storage.kuzu.db_path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/] kuzu database directory: {db_path}")

    # Apply migrations against the configured backend.
    from contextd.storage.factory import build_graph_store

    store = build_graph_store(cfg)
    store.connect()
    try:
        if cfg.storage.backend == "memgraph":
            from contextd.migrations.memgraph import ALL_MIGRATIONS

            store.apply_migrations(ALL_MIGRATIONS)
        else:
            from contextd.migrations.kuzu import ALL_MIGRATIONS

            store.apply_migrations(ALL_MIGRATIONS)
        console.print("[green]✓[/] migrations applied")
    finally:
        store.close()
    console.print("[bold]ready[/]")


@cli.command()
def down() -> None:
    """Stop the storage backend and indexer."""
    cfg = _load_cfg()
    if cfg.storage.backend == "memgraph":
        compose_file = Path(cfg.storage.memgraph.docker_compose_file).expanduser()
        subprocess.run(["docker", "compose", "-f", str(compose_file), "down"], check=False)
    console.print("[green]✓[/] stopped")


@cli.command()
def status() -> None:
    """Report daemon + backend + corpora state."""
    cfg = _load_cfg()
    console.print(f"[bold]backend:[/] {cfg.storage.backend}")
    corpora_dir = contextd_home() / "corpora"
    if corpora_dir.exists():
        corpora = list(corpora_dir.glob("*.toml"))
        console.print(f"[bold]corpora:[/] {len(corpora)} registered")
        for c in corpora:
            console.print(f"  - {c.stem}")
    else:
        console.print("[bold]corpora:[/] none (run `contextd init`)")
