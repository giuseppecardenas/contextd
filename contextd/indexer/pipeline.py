"""Coordinates bootstrap + incremental indexing flow."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from contextd.corpus_config import CorpusConfig
from contextd.indexer import phases
from contextd.indexer.hasher import FileHasher
from contextd.inference.relate import RelationshipInferrer
from contextd.inference.summarise import Summariser
from contextd.providers.base import EmbeddingProvider
from contextd.storage.base import GraphStore

RefreshScope = Literal["inferred", "summaries", "llm", "all"]


@dataclass
class BootstrapResult:
    phases: list[phases.PhaseResult]


_DEFAULT_EXCLUDE_DIRS = frozenset({".git", ".venv", "__pycache__", "node_modules"})


def _partition_markdown(files: list[Path]) -> tuple[list[Path], list[Path]]:
    """Split *files* into (markdown, non_markdown) by suffix.

    Used in section-granular mode to route .md files through the
    section-level phase pipeline and all other files through the
    file-granular phase pipeline.  Order within each bucket is preserved
    so that phase stat counts are deterministic.
    """
    md: list[Path] = []
    other: list[Path] = []
    for f in files:
        if f.suffix == ".md":
            md.append(f)
        else:
            other.append(f)
    return md, other


def enumerate_corpus_files(corpus: CorpusConfig) -> list[Path]:
    """Expand the corpus's include globs into a list of files.

    Defence against accidental walk-into-.git / node_modules / venv:
    any file whose path contains a `_DEFAULT_EXCLUDE_DIRS` component
    is dropped unless the user's `include` glob names it explicitly.
    Symlinks are skipped to avoid cycles. Users who actually need
    those paths indexed can still use an explicit glob pattern.
    """
    root = Path(corpus.corpus.root).expanduser()
    hits: list[Path] = []
    for pattern in corpus.corpus.include:
        hits.extend(root.glob(pattern))
    excl = {root / e for e in corpus.corpus.exclude}

    def _allowed(p: Path) -> bool:
        if p in excl or not p.is_file() or p.is_symlink():
            return False
        # Drop anything under a conventional exclude directory unless the
        # include glob explicitly named that directory in its path prefix.
        return not any(part in _DEFAULT_EXCLUDE_DIRS for part in p.parts)

    return [p for p in hits if _allowed(p)]


def _wipe_for_refresh(corpus: CorpusConfig, store: GraphStore, scope: RefreshScope) -> None:
    """Wipe a dependency layer of the corpus before bootstrap re-fills it.

    Scopes:
      - inferred:  DELETE origin='inferred' edges + REMOVE inferred_at markers.
      - summaries: REMOVE summary + key_points + summary_confidence on nodes.
      - llm:       union of inferred + summaries.
      - all:       DETACH DELETE Section + File + Corpus for this corpus.
                   Cascades to all attached edges (structural, inferred, manual).

    REMOVE on a missing property is a no-op, so the helper is safe on corpora
    at any state. Target-stub nodes created by relate (Pattern/Artifact/etc.
    lacking a `corpus` property) are not touched by any scope — they may be
    orphaned after `inferred` / `all` and will be gc'd by a future
    --gc-orphans path, not by this helper.
    """
    c = corpus.corpus.name
    if scope in ("inferred", "llm"):
        store.exec_write(
            "MATCH (a {corpus: $c})-[r]->() WHERE r.origin = 'inferred' DELETE r",
            {"c": c},
        )
        store.exec_write(
            "MATCH (n:Section {corpus: $c}) REMOVE n.inferred_at",
            {"c": c},
        )
        store.exec_write(
            "MATCH (n:File {corpus: $c}) REMOVE n.inferred_at",
            {"c": c},
        )
    if scope in ("summaries", "llm"):
        store.exec_write(
            "MATCH (n:Section {corpus: $c}) REMOVE n.summary, n.key_points, n.summary_confidence",
            {"c": c},
        )
        store.exec_write(
            "MATCH (n:File {corpus: $c}) REMOVE n.summary, n.key_points, n.summary_confidence",
            {"c": c},
        )
    if scope == "all":
        store.exec_write("MATCH (n:Section {corpus: $c}) DETACH DELETE n", {"c": c})
        store.exec_write("MATCH (n:File {corpus: $c}) DETACH DELETE n", {"c": c})
        store.exec_write("MATCH (n:Corpus {name: $c}) DETACH DELETE n", {"c": c})


@dataclass
class IncrementalResult:
    action: str  # "indexed" | "deleted" | "skipped"
    path: str


def _clear_file_for_reindex(path: Path, store: GraphStore) -> None:
    """Wipe per-file markers so IS-NULL guards re-process the file on next pass.

    Order: delete inferred edges first (edge cleanup must precede marker
    removal so a crash between the two leaves the file in a retry-able state),
    then REMOVE inferred_at, then REMOVE summary/key_points/summary_confidence.
    REMOVE on a missing property is a no-op on both backends.
    """
    file_path = str(path)
    store.delete_edges(file_path, origin="inferred", src_label="File")
    store.exec_write(
        "MATCH (f:File {path: $path}) REMOVE f.inferred_at",
        {"path": file_path},
    )
    store.exec_write(
        "MATCH (f:File {path: $path}) REMOVE f.summary, f.key_points, f.summary_confidence",
        {"path": file_path},
    )


def _clear_sections_for_reindex(path: Path, store: GraphStore, corpus: str) -> None:
    """Wipe per-section markers for a single file (section-granular mode only)."""
    file_path = str(path)
    store.exec_write(
        "MATCH (s:Section {corpus: $corpus, path: $path})-[r]-() "
        "WHERE r.origin = 'inferred' DELETE r",
        {"corpus": corpus, "path": file_path},
    )
    store.exec_write(
        "MATCH (s:Section {corpus: $corpus, path: $path}) "
        "REMOVE s.inferred_at, s.summary, s.key_points, s.summary_confidence",
        {"corpus": corpus, "path": file_path},
    )


def _clear_section_for_reindex(section_id: str, corpus: str, store: GraphStore) -> None:
    """Wipe markers for a single section so IS-NULL guards re-process it."""
    store.exec_write(
        "MATCH (s:Section {id: $id, corpus: $corpus})-[r]-() WHERE r.origin = 'inferred' DELETE r",
        {"id": section_id, "corpus": corpus},
    )
    store.exec_write(
        "MATCH (s:Section {id: $id, corpus: $corpus}) "
        "REMOVE s.inferred_at, s.summary, s.key_points, s.summary_confidence",
        {"id": section_id, "corpus": corpus},
    )


def run_incremental_file(
    path: Path,
    corpus: CorpusConfig,
    store: GraphStore,
    hasher: FileHasher,
    embedder: EmbeddingProvider,
    summariser: Summariser,
    inferrer: RelationshipInferrer,
    entity_sampler: Callable[[GraphStore], list[str]],
    *,
    inference_concurrency: int = 1,
) -> IncrementalResult:
    """Re-index a single changed file or record its deletion.

    Deletion path: path absent → DETACH DELETE File (+ Sections for
    section-granular .md) → return action='deleted'.

    Update path: clear stale markers → run applicable phases → return
    action='indexed'. Phase functions with IS-NULL guards process only the
    cleared file; other corpus files already have their markers set and are
    skipped cheaply.
    """
    file_path = str(path)

    if not path.exists():
        if corpus.corpus.granularity == "section" and path.suffix == ".md":
            store.exec_write(
                "MATCH (f:File {path: $path})-[:CONTAINS]->(s:Section) DETACH DELETE s",
                {"path": file_path},
            )
        store.exec_write(
            "MATCH (f:File {path: $path}) DETACH DELETE f",
            {"path": file_path},
        )
        return IncrementalResult(action="deleted", path=file_path)

    _clear_file_for_reindex(path, store)

    if corpus.corpus.granularity == "section" and path.suffix == ".md":
        file_path_str = str(path)
        corpus_name = corpus.corpus.name

        # Parse current sections and compute per-section hashes
        parsed = phases._build_parser(corpus).parse(path.read_text(errors="replace"))
        current_hashes = {
            f"{file_path_str}#{sec.anchor}": hashlib.md5(
                (sec.title + "\n\n" + sec.body).encode()
            ).hexdigest()
            for sec in parsed
        }

        # Query graph for stored hashes
        rows = store.exec_read(
            "MATCH (s:Section {path: $path, corpus: $corpus}) RETURN s.id AS id, s.hash AS hash",
            {"path": file_path_str, "corpus": corpus_name},
        )
        graph_hashes: dict[str, str | None] = {r["id"]: r.get("hash") for r in rows}

        # Sections to re-process: changed, no stored hash, or new
        to_reprocess = {sid for sid, h in current_hashes.items() if graph_hashes.get(sid) != h}

        if not to_reprocess:
            return IncrementalResult(action="skipped", path=file_path_str)

        # Selective clear — only wipe markers for changed/new sections
        for sid in to_reprocess:
            _clear_section_for_reindex(sid, corpus_name, store)

        # Re-enumerate writes current hash+embedding for ALL sections;
        # IS-NULL guards protect unchanged sections in summarise/relate
        phases.phase_enumerate_sections([path], corpus, store, embedder, hasher)
        phases.gc_sections_for_file(path, corpus, store)
        phases.phase_summarise_sections(
            corpus, summariser, store, concurrency=inference_concurrency
        )
        phases.phase_relate_sections(
            corpus, inferrer, store, entity_sampler, concurrency=inference_concurrency
        )
        phases.derive_file_level_for_path(path, corpus, store)
    else:
        phases.phase_enumerate([path], corpus.corpus.name, hasher, store, embedder)
        phases.phase_summarise([path], summariser, store, concurrency=inference_concurrency)
        phases.phase_relate(
            [path],
            inferrer,
            store,
            entity_sampler,
            corpus=corpus.corpus.name,
            concurrency=inference_concurrency,
        )

    return IncrementalResult(action="indexed", path=file_path)


def run_bootstrap(
    corpus: CorpusConfig,
    store: GraphStore,
    embedder: EmbeddingProvider,
    summariser: Summariser,
    inferrer: RelationshipInferrer,
    hasher: FileHasher,
    entity_sampler: Callable[[GraphStore], list[str]],
    *,
    inference_concurrency: int = 1,
    refresh: RefreshScope | None = None,
) -> BootstrapResult:
    if refresh is not None:
        _wipe_for_refresh(corpus, store, refresh)
    files = enumerate_corpus_files(corpus)
    results: list[phases.PhaseResult] = []
    if corpus.corpus.granularity == "section":
        # Section-granular path (spec §5.11 + M10.9).
        #
        # Non-.md files cannot have sections (the heading parser yields zero
        # sections for Lua, TOML, etc.).  They are routed through the
        # file-granular phase pipeline instead so their File.summary is
        # populated and they remain searchable.
        #
        # Partition: md_files → section pipeline; other_files → file pipeline.
        md_files, other_files = _partition_markdown(files)

        # --- Section pipeline for .md files ---
        # Embedder passed to phase_enumerate_sections so that Section.embedding
        # is included at CREATE time.
        results.append(phases.phase_enumerate_sections(md_files, corpus, store, embedder, hasher))
        # SD #74: drop stale Section nodes (only .md files produce sections).
        results.append(phases.phase_gc_sections(md_files, corpus, store))
        # Accounting phase: Section embeddings written at CREATE time.
        results.append(phases.phase_embed_sections(corpus, store))
        results.append(
            phases.phase_summarise_sections(
                corpus, summariser, store, concurrency=inference_concurrency
            )
        )
        results.append(
            phases.phase_relate_sections(
                corpus, inferrer, store, entity_sampler, concurrency=inference_concurrency
            )
        )
        results.append(phases.phase_derive_file_level(corpus, store))

        # --- File-granular pipeline for non-.md files ---
        if other_files:
            results.append(
                phases.phase_enumerate(other_files, corpus.corpus.name, hasher, store, embedder)
            )
            results.append(phases.phase_embed(other_files))
            results.append(
                phases.phase_summarise(
                    other_files, summariser, store, concurrency=inference_concurrency
                )
            )
            results.append(
                phases.phase_relate(
                    other_files,
                    inferrer,
                    store,
                    entity_sampler,
                    corpus=corpus.corpus.name,
                    concurrency=inference_concurrency,
                )
            )

        results.append(phases.phase_close(corpus.corpus.name, store, results))
    else:
        # File-granular path (default, spec §5.9).
        # phase_enumerate accepts an embedder so that embedding vectors are
        # included in the initial upsert_node call (CREATE time).
        results.append(phases.phase_enumerate(files, corpus.corpus.name, hasher, store, embedder))
        # phase_embed is an accounting-only pass (embedding already done in enumerate).
        results.append(phases.phase_embed(files))
        results.append(
            phases.phase_summarise(files, summariser, store, concurrency=inference_concurrency)
        )
        results.append(
            phases.phase_relate(
                files,
                inferrer,
                store,
                entity_sampler,
                corpus=corpus.corpus.name,
                concurrency=inference_concurrency,
            )
        )
        results.append(phases.phase_close(corpus.corpus.name, store, results))
    return BootstrapResult(phases=results)
