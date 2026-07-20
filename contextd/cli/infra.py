"""Infra-management commands: ``up`` / ``down`` / ``status``."""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import click
import psutil

from contextd._compat import (
    connect_ipc,
    daemon_popen_kwargs,
    ipc_file_name,
    process_is_alive,
)
from contextd._paths import contextd_home
from contextd.cli import cli
from contextd.cli._shared import _load_cfg, console

if TYPE_CHECKING:
    from contextd.config import Config
    from contextd.storage.base import GraphStore


# ---------------------------------------------------------------------------
# Daemon lifecycle helpers
# ---------------------------------------------------------------------------


def _pid_path() -> Path:
    return contextd_home() / "state" / "indexer.pid"


def _query_ipc_status() -> dict[str, object] | None:
    """Try to read richer daemon state via the IPC endpoint.

    Returns the parsed status dict on success, or None if the endpoint
    file is absent, connection is refused, or the round-trip takes
    longer than 1s.
    """
    ipc_path = contextd_home() / ipc_file_name()
    if not ipc_path.exists():
        return None
    try:
        with connect_ipc(ipc_path) as s:
            s.settimeout(1.0)
            s.sendall((json.dumps({"cmd": "status"}) + "\n").encode())
            raw = s.recv(4096).decode().strip()
        return dict(json.loads(raw))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None


def _write_pid(pid_path: Path, pid: int) -> None:
    pid_path.write_text(str(pid))


def _daemon_pid() -> int | None:
    try:
        return int(_pid_path().read_text().strip())
    except (OSError, ValueError):
        return None


def _daemon_is_running(pid: int) -> bool:
    return process_is_alive(pid)


def _request_daemon_stop(ipc_path: Path) -> bool:
    """Send ``{"cmd": "stop"}`` to the daemon's IPC endpoint.

    Returns True iff the daemon ack'd with ``{"ok": true}``. False covers
    both transport failures (endpoint missing, port file unreadable, refused
    connection) and protocol failures (malformed reply).
    """
    if not ipc_path.exists():
        return False
    try:
        with connect_ipc(ipc_path) as s:
            s.settimeout(1.0)
            s.sendall((json.dumps({"cmd": "stop"}) + "\n").encode())
            raw = s.recv(4096).decode().strip()
        response = json.loads(raw)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(response, dict):
        return False
    return response.get("ok") is True


def _wait_for_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_is_alive(pid):
            return True
        time.sleep(0.1)
    return not process_is_alive(pid)


def _kill_process_tree(pid: int, timeout: float) -> None:
    """Terminate the daemon process and every descendant.

    Needed on Windows where ``contextd-indexer.exe`` spawns a bootstrap
    ``python.exe`` which spawns the actual daemon ``python.exe``; killing
    the root alone leaves orphans. Sends ``terminate()`` first (POSIX:
    SIGTERM, Windows: TerminateProcess), waits, then ``kill()``s any
    survivors. Idempotent and safe if the root has already exited.
    """
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        descendants = root.children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    procs = [root, *descendants]
    for p in procs:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            p.terminate()
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            p.kill()
    if alive:
        psutil.wait_procs(alive, timeout=timeout)


def _stop_daemon() -> None:
    pid = _daemon_pid()
    if pid is None:
        return
    ipc_path = contextd_home() / ipc_file_name()
    try:
        if _request_daemon_stop(ipc_path) and _wait_for_exit(pid, timeout=5.0):
            return
        _kill_process_tree(pid, timeout=5.0)
    finally:
        _pid_path().unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            ipc_path.unlink()


def _compose_file_for(cfg: Config) -> Path:
    """Resolve the docker-compose.yml path from the Neo4j backend's config.

    Honours the ``docker_compose_file`` field on ``Neo4jConfig`` so a user
    ``config.toml`` override is respected. The default value points at
    ``~/.contextd/docker-compose.yml`` which is what ``contextd init`` deploys.
    """
    return Path(cfg.storage.neo4j.docker_compose_file).expanduser()


def _run_compose(cfg: Config, *args: str) -> None:
    """Run ``docker compose`` for the configured backend profile.

    Resolves the compose file from ``cfg`` and invokes ``docker compose -f
    <file> --profile <backend> <args...>`` with ``check=False`` so an
    already-stopped or missing container is not treated as a fatal error by the
    lifecycle commands.

    :param cfg: Loaded global config; supplies the backend name and the
        compose-file path.
    :param args: The compose subcommand and its flags, for example
        ``"stop"`` or ``"down", "--volumes"``.
    """
    backend = cfg.storage.backend
    compose_file = _compose_file_for(cfg)
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "--profile", backend, *args],
        check=False,
    )


def _wait_for_backend_ready(store: GraphStore, timeout: float = 60.0) -> None:
    # `docker compose up -d` returns when the container starts, but Neo4j
    # needs an additional ~10-60s on cold start before Bolt accepts a
    # handshake. Without this probe, the very next exec_read (inside
    # MigrationRunner) races the backend and fails with ServiceUnavailable.
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    notified = False
    while time.monotonic() < deadline:
        try:
            store.exec_read("RETURN 1 AS ok", None)
            return
        except Exception as exc:
            last_error = exc
            if not notified:
                console.print("[dim]waiting for backend to accept connections...[/]")
                notified = True
            time.sleep(1.0)
    raise click.ClickException(f"backend did not become ready within {timeout:.0f}s: {last_error}")


@cli.command()
def up() -> None:
    """Start the storage backend container and apply pending migrations."""
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
        _wait_for_backend_ready(store)
        from contextd.migrations.neo4j import ALL_MIGRATIONS

        store.apply_migrations(ALL_MIGRATIONS)
        console.print("[green]✓[/] migrations applied")
    finally:
        store.close()

    # Launch the incremental indexer daemon.
    state_dir = contextd_home() / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Guard against double-launch: if the PID file names a live process,
    # skip the Popen. If it names a dead one, clear the stale file and
    # proceed.
    existing_pid = _daemon_pid()
    if existing_pid is not None:
        if _daemon_is_running(existing_pid):
            console.print(
                f"[yellow]![/] indexer daemon already running (pid={existing_pid}); skipping launch"
            )
            console.print("[bold]ready[/]")
            return
        _pid_path().unlink(missing_ok=True)

    proc = subprocess.Popen(
        ["contextd-indexer"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **daemon_popen_kwargs(),  # type: ignore[call-overload]
    )
    _pid_path().write_text(str(proc.pid))
    console.print(f"[green]✓[/] indexer daemon launched (pid={proc.pid})")
    console.print("[bold]ready[/]")


@cli.command()
def down() -> None:
    """Stop the storage backend container and indexer, preserving all data.

    Stops (does not remove) the backend container via ``docker compose stop``,
    so the Docker data volumes and the indexed knowledge graph survive; a later
    ``contextd up`` restarts the same container with its data intact. To
    permanently delete the indexed data, use ``contextd reset``.
    """
    _stop_daemon()
    console.print("[green]✓[/] indexer daemon stopped")
    _run_compose(_load_cfg(), "stop")
    console.print("[green]✓[/] backend stopped (data preserved)")


@cli.command()
def reset() -> None:
    """Stop everything and permanently delete all indexed data.

    Removes the storage backend container together with its Docker data volumes
    (``contextd_neo4j_data`` / ``contextd_neo4j_logs``) via ``docker compose
    down --volumes``. This destroys the entire knowledge graph: every node,
    edge, summary, and embedding contextd has indexed. The deletion is
    irreversible; the only way to recover is to re-index each corpus from
    source with ``contextd index <corpus> --bootstrap``.

    Corpus registrations under ``~/.contextd/corpora/`` and ``config.toml`` are
    left untouched, so a subsequent ``contextd up`` brings up a fresh, empty
    backend ready for re-indexing. Use ``contextd down`` instead when you only
    want to stop the backend and keep the indexed data.
    """
    console.print(
        "[bold red]⚠ contextd reset will permanently delete all indexed "
        "data[/] (the knowledge graph, summaries, and embeddings in the backend "
        "volumes). This cannot be undone."
    )
    _stop_daemon()
    console.print("[green]✓[/] indexer daemon stopped")
    _run_compose(_load_cfg(), "down", "--volumes")
    console.print("[green]✓[/] backend container and data volumes removed")
    console.print("[dim]run `contextd up` to start a fresh, empty backend[/]")


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
    ipc_status = _query_ipc_status()
    if ipc_status is not None:
        ipc_pid = ipc_status.get("pid")
        ipc_uptime = ipc_status.get("uptime_seconds")
        ipc_corpora = ipc_status.get("corpora", [])
        console.print(
            f"[bold]daemon:[/] running "
            f"(pid={ipc_pid}, uptime={ipc_uptime}s, corpora={ipc_corpora!r})"
        )
    else:
        pid = _daemon_pid()
        if pid is not None and _daemon_is_running(pid):
            console.print(f"[bold]daemon:[/] running (pid={pid})")
        else:
            console.print("[bold]daemon:[/] not running")
            if pid is not None:
                _pid_path().unlink(missing_ok=True)
