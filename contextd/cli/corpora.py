"""Corpus-management commands: ``add-corpus`` / ``list-corpora`` / ``index``."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from contextd._paths import contextd_home
from contextd.cli import cli
from contextd.cli._shared import PipelineDeps, _load_cfg, console

if TYPE_CHECKING:
    from contextd.config import Config
    from contextd.corpus_config import CorpusConfig


@cli.command("add-corpus")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--name", default=None, help="Corpus name; defaults to directory basename.")
@click.option("--granularity", type=click.Choice(["file", "section"]), default="file")
@click.option(
    "--from",
    "template",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Use TEMPLATE as the TOML shape; only corpus.root (and name) are overridden.",
)
def add_corpus(path: Path, name: str | None, granularity: str, template: Path | None) -> None:
    """Register a corpus for indexing."""
    import tomli_w

    corpora_dir = contextd_home() / "corpora"
    corpora_dir.mkdir(parents=True, exist_ok=True)
    resolved_name = name or path.resolve().name
    corpus_toml = corpora_dir / f"{resolved_name}.toml"
    if corpus_toml.exists():
        console.print(f"[yellow]⚠[/] corpus {resolved_name!r} already registered at {corpus_toml}")
        return

    if template is not None:
        _add_corpus_from_template(
            path=path,
            resolved_name=resolved_name,
            corpus_toml=corpus_toml,
            template=template,
            granularity=granularity,
        )
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


def _add_corpus_from_template(
    path: Path,
    resolved_name: str,
    corpus_toml: Path,
    template: Path,
    granularity: str,
) -> None:
    """Handle `add-corpus --from TEMPLATE` — validates, rewrites, and writes the corpus TOML."""
    import tomli_w

    from contextd.corpus_config import CorpusConfig, CorpusConfigError

    # Step 1: validate via CorpusConfig.load (catches both bad TOML and bad schema).
    # CorpusConfig.load wraps pydantic errors in CorpusConfigError but lets
    # tomllib.TOMLDecodeError propagate unwrapped — catch both.
    try:
        template_cfg = CorpusConfig.load(template)
    except (CorpusConfigError, tomllib.TOMLDecodeError) as exc:
        raise click.ClickException(f"template invalid: {exc}") from exc

    # Step 2: load the raw dict for mutation (preserves unknown keys, avoids pydantic coercion).
    raw: dict[str, Any] = tomllib.loads(template.read_text(encoding="utf-8"))
    corpus_section = raw.setdefault("corpus", {})
    corpus_section["name"] = resolved_name
    corpus_section["root"] = str(path.resolve())

    # Step 3: rewrite relative paths to absolute (relative to the template's directory).
    def _rewrite(p: str) -> str:
        maybe_path = Path(p)
        if maybe_path.is_absolute():
            return str(maybe_path)
        return str((template.parent / maybe_path).resolve())

    if "ontology" in raw and "overrides" in raw["ontology"]:
        raw["ontology"]["overrides"] = _rewrite(str(raw["ontology"]["overrides"]))
    if "summarization" in raw and "prompt_override" in raw["summarization"]:
        raw["summarization"]["prompt_override"] = _rewrite(
            str(raw["summarization"]["prompt_override"])
        )
    if "mcp" in raw and "tools" in raw["mcp"]:
        raw["mcp"]["tools"] = {k: _rewrite(str(v)) for k, v in raw["mcp"]["tools"].items()}

    corpus_toml.write_bytes(tomli_w.dumps(raw).encode())

    # Granularity comes from the template; report it accurately.
    effective_granularity = template_cfg.corpus.granularity
    console.print(f"[green]✓[/] registered corpus {resolved_name!r} at {corpus_toml}")
    console.print(f"  root: {path.resolve()}")
    console.print(f"  granularity: {effective_granularity} (from template)")
    console.print(f"  template: {template.resolve()}")
    console.print(f"\n[bold]next:[/] `contextd index {resolved_name} --bootstrap`")


@cli.command("list-corpora")
def list_corpora() -> None:
    """List registered corpora."""
    corpora_dir = contextd_home() / "corpora"
    if not corpora_dir.exists():
        console.print("no corpora registered (run `contextd init` first).")
        return
    corpora = sorted(corpora_dir.glob("*.toml"))
    if not corpora:
        console.print("no corpora registered yet.")
        return
    for c in corpora:
        console.print(f"- {c.stem} ({c})")


def _build_pipeline_deps(
    cfg: Config,
    corpus_cfg: CorpusConfig,
    corpus_name: str,
    corpus_toml_path: Path,
) -> PipelineDeps:
    """Wire up the five collaborators the pipeline needs, honouring the
    corpus→global→default override hierarchy for summary length.

    ``corpus_toml_path`` is used to resolve relative ``[ontology] overrides``
    paths — relative paths resolve relative to the TOML file's parent directory.
    """
    from contextd.indexer.hasher import FileHasher
    from contextd.inference.prompts import PromptRenderer
    from contextd.inference.relate import RelationshipInferrer
    from contextd.inference.summarise import Summariser
    from contextd.ontology.overrides import OntologyOverridesError, apply_overrides
    from contextd.ontology.schema import Ontology, OntologyError
    from contextd.providers.factory import build_embedding_provider, build_inference_provider
    from contextd.storage.factory import build_graph_store

    inference_provider = build_inference_provider(cfg)
    embedding_provider = build_embedding_provider(cfg)
    renderer = PromptRenderer(contextd_home() / "prompts")
    # Apply node-label aliases first, then overlay edge-label aliases from the
    # optional overrides file.  Order matters: with_aliases() → apply_overrides().
    ontology = Ontology.load_base().with_aliases(corpus_cfg.ontology.aliases)
    if corpus_cfg.ontology.overrides is not None:
        overrides_path = Path(corpus_cfg.ontology.overrides)
        if not overrides_path.is_absolute():
            overrides_path = corpus_toml_path.parent / overrides_path
        try:
            ontology = apply_overrides(ontology, overrides_path)
        except (OntologyOverridesError, OntologyError) as exc:
            raise click.ClickException(str(exc)) from exc
    max_words = corpus_cfg.summarization.max_words or cfg.inference.summary_max_words
    prompt_path: Path | None = None
    if corpus_cfg.summarization.prompt_override is not None:
        resolved_prompt = Path(corpus_cfg.summarization.prompt_override)
        if not resolved_prompt.is_absolute():
            resolved_prompt = corpus_toml_path.parent / resolved_prompt
        resolved_prompt = resolved_prompt.resolve()
        if not resolved_prompt.exists():
            raise click.ClickException(
                f"summarization.prompt_override file not found: {resolved_prompt}"
            )
        try:
            resolved_prompt.read_text(encoding="utf-8")
        except OSError as exc:
            raise click.ClickException(
                f"summarization.prompt_override file not readable: {resolved_prompt} ({exc})"
            ) from exc
        except UnicodeDecodeError as exc:
            raise click.ClickException(
                f"summarization.prompt_override file is not valid UTF-8: {resolved_prompt} ({exc})"
            ) from exc
        prompt_path = resolved_prompt
    return PipelineDeps(
        summariser=Summariser(
            inference_provider, renderer, max_words=max_words, prompt_path=prompt_path
        ),
        inferrer=RelationshipInferrer(inference_provider, renderer, ontology),
        hasher=FileHasher(state_path=contextd_home() / "state" / f"{corpus_name}-index-state.json"),
        embedder=embedding_provider,
        store=build_graph_store(cfg),
    )


@cli.command()
@click.argument("corpus_name")
@click.option("--bootstrap", is_flag=True)
@click.option("--incremental", is_flag=True)
@click.option("--estimate-only", is_flag=True)
def index(corpus_name: str, bootstrap: bool, incremental: bool, estimate_only: bool) -> None:
    """Run an indexing pass on the named corpus."""
    from contextd.corpus_config import CorpusConfig
    from contextd.indexer.pipeline import enumerate_corpus_files, run_bootstrap

    cfg = _load_cfg()
    corpus_toml = contextd_home() / "corpora" / f"{corpus_name}.toml"
    if not corpus_toml.exists():
        raise click.ClickException(
            f"corpus {corpus_name!r} not registered."
            f" Run `contextd add-corpus <path> --name {corpus_name}` first."
        )
    corpus_cfg = CorpusConfig.load(corpus_toml)

    files = enumerate_corpus_files(corpus_cfg)
    console.print(f"found {len(files)} files in corpus {corpus_name!r}")

    if estimate_only:
        # UTF-8 character count instead of byte count — multi-byte content
        # (em-dashes, smart quotes, non-ASCII scripts) would otherwise
        # inflate the estimate. Files are read once; this is the cost the
        # estimate advertises.
        total_chars = 0
        for p in files:
            try:
                total_chars += len(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
        est_tokens = total_chars // 4  # rough: 4 chars per token
        console.print(f"~{est_tokens} input tokens projected (2 call types per file)")
        return

    if not (bootstrap or incremental):
        raise click.ClickException("specify --bootstrap or --incremental")

    deps = _build_pipeline_deps(cfg, corpus_cfg, corpus_name, corpus_toml)
    deps.store.connect()
    try:
        if bootstrap:
            result = run_bootstrap(
                corpus=corpus_cfg,
                store=deps.store,
                embedder=deps.embedder,
                summariser=deps.summariser,
                inferrer=deps.inferrer,
                hasher=deps.hasher,
                entity_sampler=lambda _s: [],
            )
            for phase in result.phases:
                console.print(
                    f"  [green]✓[/] {phase.name}:"
                    f" processed={phase.processed} skipped={phase.skipped}"
                )
        else:
            console.print("[yellow]⚠[/] incremental mode not yet implemented in this build")
    finally:
        deps.store.close()
