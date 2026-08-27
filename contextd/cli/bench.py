"""Retrieval benchmark command: ``bench``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import click

from contextd._paths import contextd_home
from contextd.cli import cli
from contextd.cli._shared import _load_cfg, console

if TYPE_CHECKING:
    from contextd.bench.run import BenchReport
    from contextd.search.collapse import ReturnUnit
    from contextd.search.graph_expand import ExpandMode

_RETURN_UNITS: tuple[str, ...] = ("chunk", "section", "file", "auto")
_EXPAND_MODES: tuple[str, ...] = ("none", "units")


def _parse_profile_sets(values: tuple[str, ...], known: list[str]) -> list[list[str] | None]:
    """Each ``--profiles`` value is a comma list; none given → one run over all.

    Unknown names are rejected up front: ``search`` would silently return
    nothing for a profile that was never indexed, which would read as a
    retrieval failure rather than a typo.
    """
    if not values:
        return [None]
    sets: list[list[str] | None] = []
    for value in values:
        names = [n.strip() for n in value.split(",") if n.strip()]
        if not names:
            raise click.ClickException("--profiles must name at least one chunk profile")
        unknown = [n for n in names if n not in known]
        if unknown:
            raise click.ClickException(
                f"unknown chunk profile(s) {', '.join(unknown)}; "
                f"this corpus declares: {', '.join(known) or '(none)'}"
            )
        sets.append(names)
    return sets


def _compare(a_path: Path, b_path: Path) -> None:
    from contextd.bench.report import load_report, render_diff

    try:
        a_reports, b_reports = load_report(a_path), load_report(b_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if len(a_reports) != len(b_reports):
        console.print(
            f"[yellow]⚠[/] {a_path} has {len(a_reports)} report(s), {b_path} has "
            f"{len(b_reports)}; comparing the first {min(len(a_reports), len(b_reports))}"
        )
    for a, b in zip(a_reports, b_reports, strict=False):
        render_diff(a, b, console)


@cli.command()
@click.argument("corpus_name", metavar="CORPUS", required=False)
@click.option(
    "--queries",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Bench spec file (TOML). Default: <corpus root>/.contextd/bench.toml.",
)
@click.option(
    "--profiles",
    "profile_sets",
    multiple=True,
    help=(
        "Comma-separated chunk profiles to query. Repeat the option to bench several "
        "configurations side by side (e.g. --profiles fine --profiles fine,coarse). "
        "Default: every profile in the graph."
    ),
)
@click.option(
    "--return-unit",
    type=click.Choice(_RETURN_UNITS),
    default=None,
    help="Unit to collapse chunk hits to. Default: [search] return_unit from config.toml.",
)
@click.option("--k", "k", type=click.IntRange(min=1), default=5, show_default=True)
@click.option(
    "--expand",
    type=click.Choice(_EXPAND_MODES),
    default=None,
    help=(
        "Graph expansion: `units` fuses Sections/Files linked to the top hits through "
        "shared entities with the direct hits. Default: [search] expand from config.toml."
    ),
)
@click.option(
    "--graph-weight",
    type=click.FloatRange(min=0.0),
    default=None,
    help="RRF weight of the expanded rows (only with --expand units). Default: [search] graph_weight.",
)
@click.option(
    "--json",
    "json_out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Save every configuration's report to this JSON file.",
)
@click.option(
    "--compare",
    nargs=2,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    metavar="A.json B.json",
    help="Diff two saved reports and exit without running anything.",
)
def bench(
    corpus_name: str | None,
    queries: Path | None,
    profile_sets: tuple[str, ...],
    return_unit: str | None,
    k: int,
    expand: str | None,
    graph_weight: float | None,
    json_out: Path | None,
    compare: tuple[Path, Path] | None,
) -> None:
    """Score retrieval (recall/precision/MRR/IoU) against a labelled query set."""
    if compare is not None:
        _compare(compare[0], compare[1])
        return
    if not corpus_name:
        raise click.ClickException("CORPUS is required unless --compare is given")

    from contextd.bench.report import render_table, save_report
    from contextd.bench.run import run_bench
    from contextd.bench.spec import BenchSpecError, load_spec
    from contextd.corpus_config import CorpusConfig, CorpusConfigError
    from contextd.providers.factory import build_embedding_provider
    from contextd.storage.factory import build_graph_store

    corpus_toml = contextd_home() / "corpora" / f"{corpus_name}.toml"
    if not corpus_toml.exists():
        raise click.ClickException(
            f"corpus {corpus_name!r} not registered."
            f" Run `contextd add-corpus <path> --name {corpus_name}` first."
        )
    try:
        corpus_cfg = CorpusConfig.load(corpus_toml)
    except CorpusConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    root = Path(corpus_cfg.corpus.root)
    if not root.is_absolute():
        root = corpus_toml.parent / root
    spec_path = queries if queries is not None else root / ".contextd" / "bench.toml"
    try:
        spec = load_spec(spec_path)
    except BenchSpecError as exc:
        raise click.ClickException(str(exc)) from exc

    known = [p.name for p in corpus_cfg.chunking.profiles]
    weights = {p.name: p.weight for p in corpus_cfg.chunking.profiles}
    sets = _parse_profile_sets(profile_sets, known)

    cfg = _load_cfg()
    unit = cast("ReturnUnit", return_unit or cfg.search.return_unit)

    embedder = None
    try:
        embedder = build_embedding_provider(cfg)
    except Exception as exc:
        console.print(f"[yellow]⚠[/] embedding provider unavailable ({exc}); full-text only")

    store = build_graph_store(cfg)
    store.connect()
    reports: list[BenchReport] = []
    try:
        for profiles in sets:
            try:
                reports.append(
                    run_bench(
                        store,
                        spec,
                        embedder=embedder,
                        search_cfg=cfg.search,
                        corpus=corpus_name,
                        profiles=profiles,
                        return_unit=unit,
                        k=k,
                        profile_weights=weights,
                        expand=cast("ExpandMode | None", expand),
                        graph_weight=graph_weight,
                    )
                )
            except Exception as exc:
                raise click.ClickException(f"bench failed: {exc}") from exc
    finally:
        store.close()

    console.print(f"[dim]spec:[/] {spec_path} ({len(spec.queries)} queries)")
    render_table(reports, console)
    if json_out is not None:
        save_report(reports, json_out)
        console.print(f"[green]✓[/] saved {len(reports)} report(s) to {json_out}")
