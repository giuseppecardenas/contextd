"""Contextd CLI.

All commands route through the global config (~/.contextd/config.toml).
The factory layer decides whether to stand up Memgraph or Kuzu based on
[storage] backend.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console

if TYPE_CHECKING:
    from contextd.config import Config

CONTEXTD_HOME = Path(os.environ.get("CONTEXTD_HOME", str(Path.home() / ".contextd")))
console = Console()


def _load_cfg() -> Config:
    """Load user config.toml with fallback to packaged default."""
    from contextd.config import Config

    path = CONTEXTD_HOME / "config.toml"
    return Config.load(path) if path.exists() else Config.load_default()


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

    copied = 0
    for name in ("summarise.md", "relate.md", "translate.md"):
        dst = home / "prompts" / name
        if not dst.exists():
            src_text = resources.files("prompts").joinpath(name).read_text()
            dst.write_text(src_text)
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

    # Docker check (informational only — KuzuDB backend doesn't need it).
    if not shutil.which("docker"):
        console.print(
            "[yellow]⚠[/] docker not on PATH; set [storage] backend = 'kuzu' to run without Docker"
        )

    console.print("\n[bold]next:[/] `contextd up` to start the daemon.")


@cli.command()
def up() -> None:
    """Start the storage backend and the indexer daemon."""
    cfg = _load_cfg()

    if cfg.storage.backend == "memgraph":
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
    corpora_dir = CONTEXTD_HOME / "corpora"
    if corpora_dir.exists():
        corpora = list(corpora_dir.glob("*.toml"))
        console.print(f"[bold]corpora:[/] {len(corpora)} registered")
        for c in corpora:
            console.print(f"  - {c.stem}")
    else:
        console.print("[bold]corpora:[/] none (run `contextd init`)")


@cli.command("add-corpus")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--name", default=None, help="Corpus name; defaults to directory basename.")
@click.option("--granularity", type=click.Choice(["file", "section"]), default="file")
def add_corpus(path: Path, name: str | None, granularity: str) -> None:
    """Register a corpus for indexing."""
    import tomli_w

    corpora_dir = CONTEXTD_HOME / "corpora"
    corpora_dir.mkdir(parents=True, exist_ok=True)
    resolved_name = name or path.resolve().name
    corpus_toml = corpora_dir / f"{resolved_name}.toml"
    if corpus_toml.exists():
        console.print(f"[yellow]⚠[/] corpus {resolved_name!r} already registered at {corpus_toml}")
        return
    data: dict[str, object] = {
        "corpus": {
            "name": resolved_name,
            "root": str(path.resolve()),
            "include": ["**/*.md"],
            "granularity": granularity,
        },
    }
    if granularity == "section":
        assert isinstance(data["corpus"], dict)
        data["corpus"]["heading_min_level"] = 2
        data["corpus"]["heading_max_level"] = 4
    corpus_toml.write_bytes(tomli_w.dumps(data).encode())
    console.print(f"[green]✓[/] registered corpus {resolved_name!r} at {corpus_toml}")
    console.print(f"  root: {path.resolve()}")
    console.print(f"  granularity: {granularity}")
    console.print(f"\n[bold]next:[/] `contextd index {resolved_name} --bootstrap`")


@cli.command("list-corpora")
def list_corpora() -> None:
    """List registered corpora."""
    corpora_dir = CONTEXTD_HOME / "corpora"
    if not corpora_dir.exists():
        console.print("no corpora registered (run `contextd init` first).")
        return
    corpora = sorted(corpora_dir.glob("*.toml"))
    if not corpora:
        console.print("no corpora registered yet.")
        return
    for c in corpora:
        console.print(f"- {c.stem} ({c})")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
