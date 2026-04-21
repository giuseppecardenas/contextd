"""Infra-management commands: ``up`` / ``down`` / ``status``."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import click

from contextd._paths import contextd_home
from contextd.cli import cli
from contextd.cli._shared import _load_cfg, console

if TYPE_CHECKING:
    from contextd.config import Config


def _compose_file_for(cfg: Config) -> Path:
    """Resolve the docker-compose.yml path from the active backend's config.

    Honours the ``docker_compose_file`` field on ``MemgraphConfig`` /
    ``Neo4jConfig`` so a user ``config.toml`` override is respected. The
    default value points at ``~/.contextd/docker-compose.yml`` which is
    what ``contextd init`` deploys.
    """
    backend = cfg.storage.backend
    if backend == "memgraph":
        compose_file_str = cfg.storage.memgraph.docker_compose_file
    elif backend == "neo4j":
        compose_file_str = cfg.storage.neo4j.docker_compose_file
    else:
        raise RuntimeError(f"unexpected backend for compose dispatch: {backend!r}")
    return Path(compose_file_str).expanduser()


@cli.command()
def up() -> None:
    """Start the storage backend and the indexer daemon."""
    cfg = _load_cfg()
    backend = cfg.storage.backend

    if not shutil.which("docker"):
        raise click.ClickException("docker not on PATH. Install Docker to run contextd.")
    compose_file = _compose_file_for(cfg)
    cmd = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "--profile",
        backend,
        "up",
        "-d",
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise click.ClickException(f"docker compose up failed (exit {result.returncode})")
    console.print(f"[green]✓[/] {backend} container up at 127.0.0.1:7687")

    # Apply migrations against the configured backend.
    from contextd.storage.factory import build_graph_store

    store = build_graph_store(cfg)
    store.connect()
    try:
        if backend == "memgraph":
            from contextd.migrations.memgraph import ALL_MIGRATIONS

            store.apply_migrations(ALL_MIGRATIONS)
        elif backend == "neo4j":
            from contextd.migrations.neo4j import ALL_MIGRATIONS

            store.apply_migrations(ALL_MIGRATIONS)
        else:
            raise RuntimeError(f"unexpected backend: {backend!r}")
        console.print("[green]✓[/] migrations applied")
    finally:
        store.close()
    console.print("[bold]ready[/]")


@cli.command()
def down() -> None:
    """Stop the storage backend and indexer."""
    cfg = _load_cfg()
    backend = cfg.storage.backend
    compose_file = _compose_file_for(cfg)
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(compose_file),
            "--profile",
            backend,
            "down",
        ],
        check=False,
    )
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
