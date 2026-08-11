"""Five-phase bootstrap pipeline per spec §5.9 Step 5.

Phase 5a: enumeration       (walk corpus, hash files, create File nodes with embeddings)
Phase 5b: embedding         (accounting phase — embeddings were created in 5a; returns count)
Phase 5c: summarisation     (Gemini per-file → File.summary + key_points)
Phase 5d: relationship inf. (Gemini per-file → typed edges, wipe-and-replace inferred)
Phase 5e: corpus closure    (write Corpus singleton stats)

Embedding vectors are computed in batch during phase_enumerate and passed to
the initial upsert_node call at CREATE time. phase_embed is a named
accounting phase that reports the count without re-issuing writes, preserving
the 5-phase contract and the integration test assertion shape.

phase_enumerate_sections follows the same pattern for Section nodes: bodies
are batch-embedded upfront and vectors are included in upsert_node at CREATE
time. Structural edges (CONTAINS File→Section, PARENT_OF Section→Section,
NEXT_SIBLING Section→Section) carry ``src_label``/``dst_label`` kwargs (the
ABC requires them; see ``GraphStore.upsert_edge``).

phase_gc_sections runs after enumerate in section mode to DETACH-DELETE
Section nodes whose anchor is no longer produced by the parser (heading
renamed between re-indexes). Without this, stale Section nodes accumulate
and pollute ``describe_project``.

M10.9: non-.md files in section-granular corpora are routed through the
file-granular phase pipeline by ``run_bootstrap`` in ``pipeline.py``.
``phase_enumerate_sections`` includes a defence-in-depth guard that logs a
warning and skips any non-.md file that reaches it, preventing accidental
mis-routing by future callers.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from contextd._paths import canonical_path
from contextd.corpus_config import CorpusConfig
from contextd.indexer.hasher import FileHasher
from contextd.indexer.heading_parser import ParsedSection, _github_anchor, section_hash
from contextd.indexer.resolution import EntityCascadeResolver, Resolution, ResolutionSettings
from contextd.indexer.units import ParseCache, extractor_for
from contextd.inference.context import CandidateRetriever, UnitIdentity
from contextd.inference.relate import InferredRelationship, RelationshipInferrer
from contextd.inference.summarise import Summariser
from contextd.ontology.schema import ENUMERATION_OWNED_LABELS, NON_ENTITY_LABELS, Ontology
from contextd.providers.base import EmbeddingProvider
from contextd.storage._keys import primary_key_for
from contextd.storage.base import GraphStore

_log = logging.getLogger(__name__)

_T = TypeVar("_T")

# Loaded once for the triple-constraint gate in _apply_inferred_edge. The
# constraint table is base-ontology-level (per-corpus configs alias names but
# never define constraints), and edge/node types arriving here are already
# alias-resolved to canon, so the base table applies uniformly.
_BASE_ONTOLOGY = Ontology.load_base()


@dataclass
class PhaseResult:
    name: str
    processed: int
    skipped: int


@dataclass
class RelateDeps:
    """Dependencies of the relate phases beyond the store.

    Grows as the resolution pipeline lands (lexical registry next); the
    optional fields default to ``None`` so direct construction in tests
    stays light — production wiring (``_build_pipeline_deps``) supplies
    everything. Constructed once and threaded through ``run_bootstrap`` /
    ``run_incremental_file`` / the daemon.
    """

    inferrer: RelationshipInferrer
    retriever: CandidateRetriever
    resolver: EntityCascadeResolver | None = None
    settings: ResolutionSettings | None = None


def _rel_path(path: Path, root: Path) -> str:
    """Corpus-root-relative posix path for prompts and routing; canonical
    absolute path when the file is outside the root (defensive fallback)."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return canonical_path(path)


def _parallel_map(
    items: Sequence[_T],
    worker: Callable[[_T], tuple[int, int]],
    concurrency: int,
) -> tuple[int, int]:
    """Run ``worker`` over ``items`` and sum the ``(processed, skipped)`` deltas.

    When ``concurrency <= 1`` the iteration is sequential so call ordering
    is preserved (matters for tests that assert on mock call order).
    When ``concurrency > 1`` workers run in a ``ThreadPoolExecutor`` — the
    inference-bound phases are I/O dominated (one HTTP round-trip to Gemini
    per item) and both graph backends declare ``concurrent_writers=-1``, so
    store writes in worker bodies are safe.

    Exceptions from the inference call must be caught inside the worker
    (matching the pre-existing "LLM error → skip, store error → fatal"
    semantics); anything that escapes the worker here propagates.
    """
    processed = skipped = 0
    if concurrency <= 1:
        for item in items:
            p, s = worker(item)
            processed += p
            skipped += s
        return processed, skipped
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, item) for item in items]
        for fut in as_completed(futures):
            p, s = fut.result()
            processed += p
            skipped += s
    return processed, skipped


def phase_enumerate(
    files: list[Path],
    corpus: str,
    hasher: FileHasher,
    store: GraphStore,
    embedder: EmbeddingProvider,
    batch_size: int = 128,
) -> PhaseResult:
    """Create File nodes with embeddings included at creation time.

    Embedder is accepted so that embedding vectors are passed to
    ``upsert_node`` at CREATE time. The phase_embed step below is a
    count-only accounting pass.
    """
    # Batch-compute embeddings for all files upfront so we can include them
    # in the initial upsert_node call.
    all_embeddings: list[list[float]] = []
    for start in range(0, len(files), batch_size):
        batch = files[start : start + batch_size]
        texts = [f.read_text(encoding="utf-8", errors="replace") for f in batch]
        all_embeddings.extend(embedder.embed(texts))

    now = dt.datetime.now(dt.UTC)
    processed = 0
    for f, vec in zip(files, all_embeddings, strict=True):
        store.upsert_node(
            "File",
            {
                "path": canonical_path(f),
                "name": f.name,
                "type": f.suffix.lstrip(".") or "unknown",
                "hash": hasher.hash(f),
                "size": f.stat().st_size,
                "corpus": corpus,
                "embedding": vec,
                "updated": now,
            },
        )
        processed += 1
    return PhaseResult(name="enumerate", processed=processed, skipped=0)


def phase_embed(files: list[Path]) -> PhaseResult:
    """Accounting phase: embedding was performed in phase_enumerate.

    Reports the count of files that were embedded, preserving the
    5-phase contract and the integration test's phases[1].processed
    assertion.
    """
    return PhaseResult(name="embed", processed=len(files), skipped=0)


def phase_summarise(
    files: list[Path],
    summariser: Summariser,
    store: GraphStore,
    *,
    concurrency: int = 1,
    corpus_cfg: CorpusConfig | None = None,
) -> PhaseResult:
    # Idempotent resume: skip files whose File node already has a summary.
    # One batch lookup against the store; set-subtracted from the input list.
    if files:
        already = {
            r["path"]
            for r in store.exec_read(
                "MATCH (f:File) WHERE f.path IN $paths AND f.summary IS NOT NULL "
                "RETURN f.path AS path",
                {"paths": [canonical_path(f) for f in files]},
            )
        }
        files = [f for f in files if canonical_path(f) not in already]

    def _identity(f: Path) -> UnitIdentity | None:
        if corpus_cfg is None:
            return None
        file_path = canonical_path(f)
        return UnitIdentity(
            corpus=corpus_cfg.corpus.name,
            file_path=file_path,
            rel_path=_rel_path(f, Path(corpus_cfg.corpus.root)),
            suffix=f.suffix,
            src_label="File",
            src_id=file_path,
        )

    def _worker(f: Path) -> tuple[int, int]:
        try:
            result = summariser.summarise(
                f.read_text(encoding="utf-8", errors="replace"), context=_identity(f)
            )
        except Exception as exc:
            # Must be logged, not just counted. The caller reports this file as
            # indexed regardless (its node and embedding do exist), so without a
            # line here a rate limit, safety block, or malformed provider
            # response leaves a permanently unsummarised file that is invisible
            # to summary search, with nothing anywhere recording why.
            _log.warning(
                "summarise failed for %s: %s: %s; File node left without a summary",
                f,
                type(exc).__name__,
                exc,
            )
            return (0, 1)
        store.exec_write(
            "MATCH (n:File {path: $path}) "
            "SET n.summary = $summary, n.key_points = $key_points, "
            "n.entities_mentioned = $entities_mentioned, "
            "n.summary_generated_at = datetime()",
            {
                "path": canonical_path(f),
                "summary": result.summary,
                "key_points": result.key_points,
                "entities_mentioned": result.entities_mentioned,
            },
        )
        return (1, 0)

    processed, skipped = _parallel_map(files, _worker, concurrency)
    return PhaseResult(name="summarise", processed=processed, skipped=skipped)


def phase_relate(
    files: list[Path],
    relate: RelateDeps,
    store: GraphStore,
    *,
    corpus_cfg: CorpusConfig,
    concurrency: int = 1,
) -> PhaseResult:
    # Idempotent resume: skip files whose File node carries an inferred_at
    # marker (set by a prior successful relate pass). Zero-edge sections are
    # still marked, so they are not re-attempted on every restart.
    if files:
        already = {
            r["path"]
            for r in store.exec_read(
                "MATCH (f:File) WHERE f.path IN $paths AND f.inferred_at IS NOT NULL "
                "RETURN f.path AS path",
                {"paths": [canonical_path(f) for f in files]},
            )
        }
        files = [f for f in files if canonical_path(f) not in already]

    corpus = corpus_cfg.corpus.name
    root = Path(corpus_cfg.corpus.root)

    def _worker(f: Path) -> tuple[int, int]:
        file_path = canonical_path(f)
        identity = UnitIdentity(
            corpus=corpus,
            file_path=file_path,
            rel_path=_rel_path(f, root),
            suffix=f.suffix,
            src_label="File",
            src_id=file_path,
        )
        try:
            # Candidates are retrieved per unit, inside the worker — the whole
            # point of the retriever seam (a phase-global sample cannot offer
            # unit-relevant targets).
            candidates = relate.retriever.for_unit(store, identity=identity)
            relations = relate.inferrer.infer(
                f.read_text(encoding="utf-8", errors="replace"),
                identity=identity,
                candidates=candidates,
            )
        except Exception as exc:
            # Logged for the same reason as the summarise failure above, plus one
            # of its own: the inferred_at marker below is what makes resume
            # idempotent, so a failure here means this file is re-inferred on
            # every subsequent pass, spending tokens indefinitely and silently.
            _log.warning(
                "relate failed for %s: %s: %s; no inferred edges written, "
                "file will be retried on the next pass",
                f,
                type(exc).__name__,
                exc,
            )
            return (0, 1)
        # Wipe-and-replace inferred edges (spec §5.5).
        # src_label="File" required by GraphStore.delete_edges (see ABC
        # docstring) — a label-less MATCH is ambiguous when endpoints
        # have non-"path" PKs.
        store.delete_edges(file_path, origin="inferred", src_label="File")
        local_skipped = 0
        for rel in relations:
            # File/Section targets resolve to an existing node or are dropped
            # (never stubbed); other labels upsert a tagged stub. See
            # _apply_inferred_edge.
            if not _apply_inferred_edge(
                store,
                file_path,
                "File",
                rel,
                corpus,
                resolver=relate.resolver,
                settings=relate.settings,
            ):
                local_skipped += 1
        # Mark processed so an interrupted run can resume without re-inferring.
        # Marker set only after the upsert loop completes; exception paths
        # return (0, 1) above and leave the marker unset.
        store.exec_write(
            "MATCH (f:File {path: $path}) SET f.inferred_at = datetime()",
            {"path": file_path},
        )
        return (1, local_skipped)

    processed, skipped = _parallel_map(files, _worker, concurrency)
    return PhaseResult(name="relate", processed=processed, skipped=skipped)


def phase_close(
    corpus: str,
    store: GraphStore,
    results: list[PhaseResult],
) -> PhaseResult:
    # SD #70: Corpus.node_count + Corpus.edge_count are persisted. Both
    # backends are schema-free at the Corpus level; the fields land directly
    # via upsert_node without DDL.
    count_files = store.exec_read(
        "MATCH (n:File {corpus: $c}) RETURN count(n) AS c", {"c": corpus}
    )[0]["c"]
    count_edges = store.exec_read("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    store.upsert_node(
        "Corpus",
        {
            "name": corpus,
            "registered_at": dt.datetime.now(dt.UTC),
            "node_count": count_files,
            "edge_count": count_edges,
        },
    )
    return PhaseResult(name="close", processed=1, skipped=0)


def phase_enumerate_sections(
    files: list[Path],
    corpus_cfg: CorpusConfig,
    store: GraphStore,
    embedder: EmbeddingProvider,
    hasher: FileHasher,
    batch_size: int = 128,
    *,
    parse_cache: ParseCache | None = None,
) -> PhaseResult:
    """Section-granular enumeration — emits Section nodes + structural edges.

    Embedder is accepted so that Section.embedding is included in
    ``upsert_node`` at CREATE time. Section bodies are batch-embedded
    upfront, then each Section is upserted with its embedding attached.

    ``upsert_edge`` calls supply ``src_label``/``dst_label`` kwargs as
    required by ``GraphStore.upsert_edge``.

    SD #73: FileHasher is threaded through so File.hash records the real
    MD5 of the file. Previously a "__pending__" sentinel blocked
    incremental re-index in section mode.
    """
    cache = parse_cache if parse_cache is not None else ParseCache(corpus_cfg)

    # Defence-in-depth (M10.9): files without a unit extractor yield zero
    # sections and would leave File.summary NULL.  The caller (run_bootstrap)
    # partitions files before calling this function; this guard catches
    # future mis-routing.
    md_files: list[Path] = []
    for f in files:
        if extractor_for(corpus_cfg, f.suffix) is None:
            _log.warning(
                "phase_enumerate_sections: skipping non-unit-parseable file %s "
                "(route through phase_enumerate instead)",
                f,
            )
        else:
            md_files.append(f)
    files = md_files

    # Collect all sections for all files first so we can batch-embed in one pass.
    parsed_by_file: list[tuple[Path, list[ParsedSection]]] = [
        (f, cache.get(f).sections) for f in files
    ]
    all_sections: list[tuple[Path, ParsedSection]] = [
        (f, sec) for f, secs in parsed_by_file for sec in secs
    ]
    all_bodies = [sec.body for _, sec in all_sections]

    # Batch-embed all section bodies.
    embeddings: list[list[float]] = []
    for start in range(0, len(all_bodies), batch_size):
        embeddings.extend(embedder.embed(all_bodies[start : start + batch_size]))

    # Canonicalise each file path once; node identity (File.path and the
    # Section.id prefix) must be derived identically here and at every
    # re-index / GC site or MERGE creates duplicates instead of updating.
    canon_path: dict[Path, str] = {f: canonical_path(f) for f, _ in parsed_by_file}

    # Build (file_path, section_id) → embedding lookup.
    embedding_map: dict[tuple[str, str], list[float]] = {
        (canon_path[f], f"{canon_path[f]}#{sec.anchor}"): vec
        for (f, sec), vec in zip(all_sections, embeddings, strict=True)
    }

    now = dt.datetime.now(dt.UTC)
    processed = 0
    for f, sections in parsed_by_file:
        file_path = canon_path[f]
        # Upsert the parent File node with a real MD5 hash (SD #73).
        store.upsert_node(
            "File",
            {
                "path": file_path,
                "name": f.name,
                "type": f.suffix.lstrip(".") or "unknown",
                "hash": hasher.hash(f),
                "size": f.stat().st_size,
                "corpus": corpus_cfg.corpus.name,
                "updated": now,
            },
        )
        previous_sibling_id: dict[str | None, str] = {}
        for sec in sections:
            section_id = f"{file_path}#{sec.anchor}"
            store.upsert_node(
                "Section",
                {
                    "id": section_id,
                    "anchor": sec.anchor,
                    "title": sec.title,
                    "level": sec.level,
                    "path": file_path,
                    "corpus": corpus_cfg.corpus.name,
                    "file_id": file_path,
                    "ordinal": sec.ordinal,
                    "embedding": embedding_map[(file_path, section_id)],
                    "hash": section_hash(sec),
                    "updated": now,
                },
            )
            # src_label/dst_label required by GraphStore.upsert_edge.
            store.upsert_edge(
                file_path,
                section_id,
                "CONTAINS",
                origin="structural",
                src_label="File",
                dst_label="Section",
            )
            if sec.parent_anchor is not None:
                parent_id = f"{file_path}#{sec.parent_anchor}"
                store.upsert_edge(
                    parent_id,
                    section_id,
                    "PARENT_OF",
                    origin="structural",
                    src_label="Section",
                    dst_label="Section",
                )
            prev = previous_sibling_id.get(sec.parent_anchor)
            if prev is not None:
                store.upsert_edge(
                    prev,
                    section_id,
                    "NEXT_SIBLING",
                    origin="structural",
                    src_label="Section",
                    dst_label="Section",
                )
            previous_sibling_id[sec.parent_anchor] = section_id
            processed += 1
    return PhaseResult(name="enumerate_sections", processed=processed, skipped=0)


def phase_gc_sections(
    files: list[Path],
    corpus_cfg: CorpusConfig,
    store: GraphStore,
    *,
    parse_cache: ParseCache | None = None,
) -> PhaseResult:
    """Delete Section nodes whose anchor is no longer produced by the parser.

    Runs after ``phase_enumerate_sections`` in section-mode bootstrap so that
    newly-created sections for the current pass are already written and will
    not be collected as stale. Builds the current-id set from parser output
    (one parse per file, cached via ``_parse_cached``), queries existing
    Section ids for the corpus, and DETACH-DELETEs the set difference. The
    DETACH DELETE cascades to both structural (CONTAINS / PARENT_OF /
    NEXT_SIBLING) and inferred (REFERENCES etc.) edges anchored at the
    stale section — no separate edge cleanup is required.

    SD #74: unblocks M11 incremental re-index. Without this phase, renaming a
    heading between re-indexes leaves the old Section node orphaned in the
    graph; ``phase_summarise_sections`` / ``phase_relate_sections`` silently
    skip such sections (their anchor is absent from the parser output) but
    the node itself persists forever and pollutes ``describe_project``.

    Per-id iteration rather than a bulk ``IN``-list parameter keeps the
    query shape simple and consistent across backends; realistic corpus
    scale (≤ a few hundred stale sections per re-index) makes the N-query
    overhead negligible.
    """
    cache = parse_cache if parse_cache is not None else ParseCache(corpus_cfg)
    current_ids: set[str] = set()
    for f in files:
        parsed = cache.get(f)
        for sec in parsed.sections:
            current_ids.add(f"{parsed.canonical}#{sec.anchor}")

    existing = store.exec_read(
        "MATCH (s:Section {corpus: $c}) RETURN s.id AS id",
        {"c": corpus_cfg.corpus.name},
    )
    stale = [r["id"] for r in existing if r["id"] not in current_ids]
    for sid in stale:
        store.exec_write(
            "MATCH (s:Section {id: $id}) DETACH DELETE s",
            {"id": sid},
        )
    return PhaseResult(name="gc_sections", processed=len(stale), skipped=0)


def gc_sections_for_file(
    path: Path,
    corpus_cfg: CorpusConfig,
    store: GraphStore,
    *,
    parse_cache: ParseCache | None = None,
) -> int:
    """GC stale Section nodes for a single file after incremental re-index.

    Runs HeadingParser on *path*, queries Section nodes for this file, and
    DETACH DELETEs any whose anchor is no longer produced by the parser
    (renamed or deleted headings). Returns the count of deleted sections.

    Called from run_incremental_file after phase_enumerate_sections so that
    renamed headings are cleaned up without waiting for the next full bootstrap.
    """
    if extractor_for(corpus_cfg, path.suffix) is None:
        return 0
    cache = parse_cache if parse_cache is not None else ParseCache(corpus_cfg)
    parsed = cache.get(path)
    file_path = parsed.canonical
    current_ids: set[str] = {f"{file_path}#{sec.anchor}" for sec in parsed.sections}
    existing = store.exec_read(
        "MATCH (s:Section {corpus: $corpus, path: $path}) RETURN s.id AS id",
        {"corpus": corpus_cfg.corpus.name, "path": file_path},
    )
    stale = [r["id"] for r in existing if r["id"] not in current_ids]
    for sid in stale:
        store.exec_write(
            "MATCH (s:Section {id: $id}) DETACH DELETE s",
            {"id": sid},
        )
    return len(stale)


def phase_embed_sections(corpus_cfg: CorpusConfig, store: GraphStore) -> PhaseResult:
    """Accounting phase: Section embeddings are written at CREATE time in
    phase_enumerate_sections. This phase counts rows and returns.

    TODO(M9-followup): if incremental re-index needs to refresh stale
    embeddings, implement a DETACH-DELETE + re-CREATE pattern here.
    """
    rows = store.exec_read(
        "MATCH (s:Section {corpus: $c}) RETURN s.id AS id",
        {"c": corpus_cfg.corpus.name},
    )
    return PhaseResult(name="embed_sections", processed=len(rows), skipped=0)


def phase_summarise_sections(
    corpus_cfg: CorpusConfig,
    summariser: Summariser,
    store: GraphStore,
    *,
    concurrency: int = 1,
    parse_cache: ParseCache | None = None,
) -> PhaseResult:
    """Summarise each Section node via LLM (spec §5.11.3).

    Reads the section body from the shared :class:`ParseCache` (one parse per
    file per pipeline invocation) and locates the section by anchor. On any
    exception (provider error, parse failure) the section is skipped and
    counted in skipped.

    Under ``concurrency > 1`` the parse cache is pre-populated serially
    before workers are dispatched; cache reads from multiple threads are
    safe, cache writes are not.
    """
    # Idempotent resume: skip Section nodes that already have a summary.
    rows = store.exec_read(
        # Path-less sections are inferred-edge target stubs (e.g. an LLM-emitted
        # target_id like "Utility Fixtures") with no parseable source file. They
        # have no body to summarise; skip them so Path(None) doesn't blow up
        # the parse-cache prefill below.
        "MATCH (s:Section {corpus: $c}) "
        "WHERE s.summary IS NULL AND s.path IS NOT NULL "
        "RETURN s.id AS id, s.path AS path",
        {"c": corpus_cfg.corpus.name},
    )
    cache = parse_cache if parse_cache is not None else ParseCache(corpus_cfg)
    for p in {Path(r["path"]) for r in rows}:
        cache.get(p)

    corpus_name = corpus_cfg.corpus.name
    root = Path(corpus_cfg.corpus.root)

    def _worker(r: dict[str, str]) -> tuple[int, int]:
        anchor = r["id"].split("#", 1)[1]
        path = Path(r["path"])
        parsed = cache.get(path)
        sec = parsed.by_anchor(anchor)
        if not sec:
            _log.debug("summarise: section %s no longer present on disk; skipping", r["id"])
            return (0, 1)
        identity = UnitIdentity(
            corpus=corpus_name,
            file_path=parsed.canonical,
            rel_path=_rel_path(path, root),
            suffix=path.suffix,
            src_label="Section",
            src_id=r["id"],
            title=sec.title,
            anchor=sec.anchor,
            parent_titles=parsed.parent_chain(sec.anchor),
        )
        try:
            result = summariser.summarise(sec.body, context=identity)
        except Exception as exc:
            _log.warning(
                "summarise failed for section %s: %s: %s; Section left without a summary",
                r["id"],
                type(exc).__name__,
                exc,
            )
            return (0, 1)
        store.exec_write(
            "MATCH (s:Section {id: $id}) "
            "SET s.summary = $summary, s.key_points = $key_points, "
            "s.entities_mentioned = $entities_mentioned, "
            "s.summary_generated_at = datetime()",
            {
                "id": r["id"],
                "summary": result.summary,
                "key_points": result.key_points,
                "entities_mentioned": result.entities_mentioned,
            },
        )
        return (1, 0)

    processed, skipped = _parallel_map(rows, _worker, concurrency)
    return PhaseResult(name="summarise_sections", processed=processed, skipped=skipped)


def phase_relate_sections(
    corpus_cfg: CorpusConfig,
    relate: RelateDeps,
    store: GraphStore,
    *,
    concurrency: int = 1,
    parse_cache: ParseCache | None = None,
) -> PhaseResult:
    """Infer typed edges from each Section node (spec §5.11.3).

    Wipe-and-replace inferred edges per section then upsert new ones.
    ``delete_edges`` and ``upsert_edge`` both supply ``src_label="Section"``
    and ``dst_label=rel.target_type`` as required by ``GraphStore``. Section
    bodies come from the shared :class:`ParseCache` (one parse per file per
    pipeline invocation).

    Under ``concurrency > 1`` the parse cache is pre-populated serially
    before workers are dispatched (see ``phase_summarise_sections``).
    """
    # Idempotent resume: skip Sections that already carry an inferred_at
    # marker. Zero-edge sections still get marked (see worker below) so
    # they are not re-attempted on every restart.
    rows = store.exec_read(
        # Same path-stub filter as phase_summarise_sections — target stubs
        # have no source file to re-parse for relate either.
        "MATCH (s:Section {corpus: $c}) "
        "WHERE s.inferred_at IS NULL AND s.path IS NOT NULL "
        "RETURN s.id AS id, s.path AS path",
        {"c": corpus_cfg.corpus.name},
    )
    cache = parse_cache if parse_cache is not None else ParseCache(corpus_cfg)
    for p in {Path(r["path"]) for r in rows}:
        cache.get(p)
    corpus_name = corpus_cfg.corpus.name
    root = Path(corpus_cfg.corpus.root)

    def _worker(r: dict[str, str]) -> tuple[int, int]:
        anchor = r["id"].split("#", 1)[1]
        path = Path(r["path"])
        parsed = cache.get(path)
        sec = parsed.by_anchor(anchor)
        if not sec:
            _log.debug("relate: section %s no longer present on disk; skipping", r["id"])
            return (0, 1)
        identity = UnitIdentity(
            corpus=corpus_name,
            file_path=parsed.canonical,
            rel_path=_rel_path(path, root),
            suffix=path.suffix,
            src_label="Section",
            src_id=r["id"],
            title=sec.title,
            anchor=sec.anchor,
            parent_titles=parsed.parent_chain(sec.anchor),
        )
        try:
            # Per-unit candidate retrieval — see phase_relate.
            candidates = relate.retriever.for_unit(store, identity=identity)
            relations = relate.inferrer.infer(sec.body, identity=identity, candidates=candidates)
        except Exception as exc:
            _log.warning(
                "relate failed for section %s: %s: %s; no inferred edges written, "
                "section will be retried on the next pass",
                r["id"],
                type(exc).__name__,
                exc,
            )
            return (0, 1)
        # Wipe-and-replace inferred edges for this section (spec §5.5).
        store.delete_edges(r["id"], origin="inferred", src_label="Section")
        local_skipped = 0
        for rel in relations:
            # File/Section targets resolve to an existing node or are dropped
            # (never stubbed); other labels upsert a tagged stub. See
            # _apply_inferred_edge.
            if not _apply_inferred_edge(
                store,
                r["id"],
                "Section",
                rel,
                corpus_name,
                resolver=relate.resolver,
                settings=relate.settings,
            ):
                local_skipped += 1
        # Mark processed so resume can skip. Only set after the upsert loop
        # completes; exception paths above return (0, 1) unmarked.
        store.exec_write(
            "MATCH (s:Section {id: $id}) SET s.inferred_at = datetime()",
            {"id": r["id"]},
        )
        return (1, local_skipped)

    processed, skipped = _parallel_map(rows, _worker, concurrency)
    return PhaseResult(name="relate_sections", processed=processed, skipped=skipped)


def phase_derive_file_level(
    corpus_cfg: CorpusConfig,
    store: GraphStore,
) -> PhaseResult:
    """Derive File.summary from child section summaries (spec §5.11.3).

    File.embedding is NOT derived in section mode — centroid computation
    is not attempted; File.embedding remains NULL in section-mode corpora.
    Callers that need a file-level embedding in section mode should
    compute a centroid at query time over the Section embeddings.
    """
    rows = store.exec_read(
        "MATCH (f:File {corpus: $c})-[:CONTAINS]->(s:Section) "
        "RETURN f.path AS path, collect(s.summary) AS summaries",
        {"c": corpus_cfg.corpus.name},
    )
    for r in rows:
        summaries = [s for s in r["summaries"] if s]
        summary = _concat_first_sentences(summaries, max_chars=500)
        store.exec_write(
            "MATCH (f:File {path: $path}) SET f.summary = $summary",
            {"path": r["path"], "summary": summary},
        )
    return PhaseResult(name="derive_file_level", processed=len(rows), skipped=0)


def derive_file_level_for_path(
    path: Path,
    corpus_cfg: CorpusConfig,
    store: GraphStore,
) -> None:
    """Derive File.summary from Section summaries for a single file.

    Queries only the sections of *path* and sets File.summary via
    _concat_first_sentences. O(1) w.r.t. corpus size — called from
    run_incremental_file instead of the full-corpus phase_derive_file_level.
    """
    file_path = canonical_path(path)
    rows = store.exec_read(
        "MATCH (f:File {path: $path})-[:CONTAINS]->(s:Section) "
        "RETURN collect(s.summary) AS summaries",
        {"path": file_path},
    )
    if not rows:
        return
    summaries = [s for s in rows[0]["summaries"] if s]
    summary = _concat_first_sentences(summaries, max_chars=500)
    store.exec_write(
        "MATCH (f:File {path: $path}) SET f.summary = $summary",
        {"path": file_path, "summary": summary},
    )


def _infer_key(target_type: str) -> str:
    """Return the primary-key property name for target_type, or raise ValueError.

    Delegates to contextd.storage._keys.primary_key_for — the authoritative
    label→PK map that mirrors the migration DDL. Unknown labels raise
    ValueError; the phase_relate / phase_relate_sections call sites catch
    and skip so a hallucinated edge target doesn't abort the whole batch.
    """
    return primary_key_for(target_type)


# File and Section nodes mirror real on-disk content and are created ONLY by
# the enumerate phases. Inference must never mint them: a stub (PK + corpus,
# no path/summary/embedding) is a phantom "old" record that pollutes
# section/file queries and is exactly what the wipe-and-replace edge logic
# leaves orphaned on the next re-inference. References to them are resolved to
# the existing node instead — or dropped. Aliased from the shared ontology
# constant so the relate phase, the parse gate, and prune-entities agree.
_ENUMERATION_OWNED_LABELS = ENUMERATION_OWNED_LABELS

# Fallback floor when no ResolutionSettings are supplied (direct phase calls
# in tests); production wiring passes per-corpus settings. The relate prompt
# documents "below 0.5 skip", but prompt rules are advisory — this is the
# enforced floor.
_DEFAULT_RESOLUTION_SETTINGS = ResolutionSettings()


_DOTTED_NUMBER = re.compile(r"\d+(\.\d+)*")


def _unique_section_id(
    store: GraphStore, corpus: str, predicate: str, params: dict[str, Any], rule: str
) -> str | None:
    """Run a Section lookup and return its id iff exactly one row matches.

    Non-unique matches return ``None`` — mirroring the File basename rule,
    ambiguity means unresolved rather than mis-linked. A unique hit is logged
    with the rule that produced it, feeding the resolution audit trail.
    """
    rows = store.exec_read(
        f"MATCH (n:Section {{corpus: $c}}) WHERE n.path IS NOT NULL AND {predicate} "
        "RETURN n.id AS v LIMIT 2",
        {"c": corpus, **params},
    )
    if len(rows) == 1:
        _log.info("relate resolve: Section matched by %s: %.120s", rule, rows[0]["v"])
        return str(rows[0]["v"])
    return None


def _resolve_section_fallback(store: GraphStore, needle: str, corpus: str) -> str | None:
    """Resolve a Section citation that is not an exact ``path#anchor`` id.

    The model cites sections as ``§12.2.5``, ``Trade Route Decay``, or
    ``some/file.md#anchor`` — never as the absolute canonical id it has no way
    to know. Fallback ladder, each rung unique-only:

      1. ``#``-fragment anchor match (a relative-path citation carries the
         right anchor even when the path half doesn't resolve),
      2. slugified-title anchor match,
      3. case-insensitive exact title match,
      4. dotted-number title prefix (``12.2.5`` → ``§12.2.5 Trade routes``).
    """
    cleaned = needle.lstrip("§").strip()
    if not cleaned:
        return None
    if "#" in cleaned:
        fragment = cleaned.rsplit("#", 1)[1]
        if fragment:
            hit = _unique_section_id(
                store, corpus, "n.anchor = $a", {"a": fragment}, "anchor fragment"
            )
            if hit is not None:
                return hit
    hit = _unique_section_id(
        store, corpus, "n.anchor = $a", {"a": _github_anchor(cleaned)}, "slugified anchor"
    )
    if hit is not None:
        return hit
    hit = _unique_section_id(
        store, corpus, "toLower(n.title) = toLower($t)", {"t": cleaned}, "title"
    )
    if hit is not None:
        return hit
    if _DOTTED_NUMBER.fullmatch(cleaned):
        # Space-suffixed prefixes so needle 12.2.5 cannot match "12.2.50 ...";
        # bare-equality arms cover a heading that is only the number.
        return _unique_section_id(
            store,
            corpus,
            "(n.title STARTS WITH ($t + ' ') OR n.title STARTS WITH ('§' + $t + ' ') "
            "OR n.title = $t OR n.title = ('§' + $t))",
            {"t": cleaned},
            "numbered-title prefix",
        )
    return None


def _resolve_existing_node(store: GraphStore, label: str, raw_name: str, corpus: str) -> str | None:
    """Resolve an inferred-edge target to an EXISTING node's primary-key value.

    Used only for ``_ENUMERATION_OWNED_LABELS`` (File/Section). Resolution is
    corpus-scoped and best-effort:

      * exact primary-key match (path separators normalised to ``/`` so an
        LLM citing ``docs\\x.md`` matches the canonical ``docs/x.md``), then
      * for ``File``, a *unique* basename match on the ``name`` property
        (the LLM commonly cites a file by bare name, e.g. ``03-economy.md``);
        a non-unique basename is left unresolved rather than mis-linked, then
      * for ``Section``, the :func:`_resolve_section_fallback` ladder
        (anchor fragment / slugified anchor / title / numbered-title prefix),
        each rung unique-only.

    Returns the matched PK value, or ``None`` when nothing real matches — the
    caller then drops the edge rather than creating a phantom stub.
    """
    pk = primary_key_for(label)
    needle = raw_name.replace("\\", "/")
    # ``label`` is constrained to _ENUMERATION_OWNED_LABELS by the sole caller,
    # so the interpolation here is not attacker-influenced.
    exact = store.exec_read(
        f"MATCH (n:{label} {{corpus: $c}}) WHERE n.{pk} = $v RETURN n.{pk} AS v LIMIT 1",
        {"c": corpus, "v": needle},
    )
    if exact:
        return str(exact[0]["v"])
    if label == "File":
        basename = needle.rsplit("/", 1)[-1]
        by_name = store.exec_read(
            "MATCH (n:File {corpus: $c}) WHERE n.name = $b AND n.hash IS NOT NULL "
            "RETURN n.path AS v LIMIT 2",
            {"c": corpus, "b": basename},
        )
        if len(by_name) == 1:
            return str(by_name[0]["v"])
    if label == "Section":
        return _resolve_section_fallback(store, needle, corpus)
    return None


def _apply_inferred_edge(
    store: GraphStore,
    src_id: str,
    src_label: str,
    rel: InferredRelationship,
    corpus: str,
    *,
    resolver: EntityCascadeResolver | None = None,
    settings: ResolutionSettings | None = None,
) -> bool:
    """Write one inferred edge from ``src_id`` to ``rel``'s target.

    Returns ``True`` if an edge was written, ``False`` if the edge was dropped.

    For enumeration-owned target labels (File/Section) the target is resolved
    to an existing node and the edge is dropped when it does not resolve —
    never stubbed. For every other target label the destination is upserted as
    a lightweight stub (legitimate abstract entities such as
    ``Pattern``/``Risk``/``Ticket`` the LLM identifies from prose), tagged with
    the current ``corpus`` so corpus-scoped GC and queries can see it.

    ``src_label``/``dst_label`` are required by ``GraphStore.upsert_edge``.
    Every drop path emits an INFO log naming the reason, so discarded edges
    are countable from the log rather than invisible.
    """
    floor = (settings or _DEFAULT_RESOLUTION_SETTINGS).confidence_floor
    if rel.confidence < floor:
        _log.info(
            "relate drop: confidence %.2f below floor %.2f: %s -[%s]-> %s(%.80s)",
            rel.confidence,
            floor,
            src_id,
            rel.edge_type,
            rel.target_type,
            rel.target_name,
        )
        return False
    if rel.target_type in NON_ENTITY_LABELS and rel.target_type not in _ENUMERATION_OWNED_LABELS:
        # Backstop for the parse-gate rule: Corpus/Meta are system labels and
        # must never be minted as inference targets, whichever path (LLM or
        # lexical) produced the relationship.
        _log.info(
            "relate drop: system label %r is not an inference target: %s -[%s]-> %.80s",
            rel.target_type,
            src_id,
            rel.edge_type,
            rel.target_name,
        )
        return False
    if not _BASE_ONTOLOGY.validate_triple(src_label, rel.edge_type, rel.target_type):
        # Combination-level gate: type-by-type checks let junk like
        # Section -DOCUMENTS-> Client through. Edge types reach this point
        # already alias-resolved to canon, so the base constraint table
        # applies uniformly regardless of per-corpus aliases.
        _log.info(
            "relate drop: triple not allowed: %s -[%s]-> %s (%.80s)",
            src_label,
            rel.edge_type,
            rel.target_type,
            rel.target_name,
        )
        return False
    try:
        pk = _infer_key(rel.target_type)
    except ValueError:
        # Hallucinated target label absent from the ontology key map.
        _log.info(
            "relate drop: unknown target label %r: %s -[%s]-> %.80s",
            rel.target_type,
            src_id,
            rel.edge_type,
            rel.target_name,
        )
        return False
    if rel.target_type in _ENUMERATION_OWNED_LABELS:
        target_value = _resolve_existing_node(store, rel.target_type, rel.target_name, corpus)
        if target_value is None:
            _log.info(
                "relate drop: %s target did not resolve: %s -[%s]-> %.80s",
                rel.target_type,
                src_id,
                rel.edge_type,
                rel.target_name,
            )
            return False
        if target_value == src_id and rel.target_type == src_label:
            _log.info(
                "relate drop: self-loop: %s -[%s]-> itself",
                src_id,
                rel.edge_type,
            )
            return False
    else:
        # Mintable entity: run the resolution cascade first — an existing node
        # with the same normalized identity absorbs the edge instead of a new
        # stub fragmenting the graph. The primary key is excluded from the
        # content merge so the resolved PK value is never overwritten by a
        # divergent model-supplied value — this also protects ``Risk``, whose
        # PK is the content field ``description``.
        if resolver is not None:
            resolution: Resolution | None = resolver.resolve(
                rel.target_type, rel.target_name, corpus
            )
        else:
            resolution = None
        target_value = resolution.pk_value if resolution is not None else rel.target_name
        props: dict[str, Any] = {pk: target_value, "corpus": corpus}
        if resolution is not None and resolution.action == "minted":
            props["name_norm"] = resolution.norm
            if resolution.vector is not None:
                props["embedding"] = resolution.vector
        for key, value in rel.target_properties.items():
            if key != pk:
                props[key] = value
        store.upsert_node(rel.target_type, props)
    store.upsert_edge(
        src_id,
        target_value,
        rel.edge_type,
        origin="inferred",
        properties={"confidence": rel.confidence, "reason": rel.reason},
        src_label=src_label,
        dst_label=rel.target_type,
    )
    return True


def _concat_first_sentences(summaries: list[str], *, max_chars: int) -> str:
    """Concatenate the first sentence of each summary up to max_chars total."""
    out: list[str] = []
    total = 0
    for s in summaries:
        sentence = s.split(".", 1)[0] + "."
        if total + len(sentence) + 1 > max_chars:
            break
        out.append(sentence)
        total += len(sentence) + 1
    return " ".join(out)
