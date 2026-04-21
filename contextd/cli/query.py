"""Query-side commands: ``ask`` / ``logs`` / ``costs``."""

from __future__ import annotations

import json
import subprocess

import click

from contextd._paths import contextd_home
from contextd.cli import cli
from contextd.cli._shared import _load_cfg, console


@cli.command()
@click.argument("question")
@click.option("--corpus", default=None)
def ask(question: str, corpus: str | None) -> None:
    """Natural-language query — translates to Cypher and runs it."""
    from contextd.inference.prompts import PromptRenderer
    from contextd.inference.translate import QueryTranslator
    from contextd.ontology.schema import Ontology
    from contextd.providers.factory import build_inference_provider
    from contextd.storage.factory import build_graph_store

    cfg = _load_cfg()
    translator = QueryTranslator(
        provider=build_inference_provider(cfg),
        renderer=PromptRenderer(contextd_home() / "prompts"),
        ontology=Ontology.load_base(),
    )
    try:
        cypher = translator.translate(question, corpus=corpus)
    except Exception as exc:
        raise click.ClickException(f"translation failed: {exc}") from exc
    console.print(f"[dim]cypher:[/] {cypher}")
    store = build_graph_store(cfg)
    store.connect()
    try:
        try:
            rows = store.exec_read(cypher, {})
        except Exception as exc:
            raise click.ClickException(f"query failed: {exc}") from exc
        console.print(json.dumps(rows, indent=2, default=str))
    finally:
        store.close()


@cli.command()
@click.option("--follow", is_flag=True)
def logs(follow: bool) -> None:
    """Tail the structured JSON log."""
    log_path = contextd_home() / "logs" / "contextd.log"
    if not log_path.exists():
        console.print(f"no log at {log_path}")
        return
    if follow:
        try:
            subprocess.run(["tail", "-f", str(log_path)], check=False)
        except KeyboardInterrupt:
            # Ctrl-C is the normal way to end `--follow`; render a clean
            # exit instead of Click's "Aborted!" output.
            console.print("")
    else:
        console.print(log_path.read_text(encoding="utf-8"))


@cli.command()
@click.option("--since", default=None, help="YYYY-MM-DD lower bound (inclusive).")
def costs(since: str | None) -> None:
    """Aggregated provider token spend."""
    from contextd.providers.cost_log import CostLog

    log = CostLog(contextd_home() / "state" / "session-log")
    totals = log.aggregate(since=since)
    if not totals:
        console.print("no usage recorded yet.")
        return
    for provider, counts in totals.items():
        console.print(
            f"[bold]{provider}:[/] input={counts['input_tokens']} output={counts['output_tokens']}"
        )
