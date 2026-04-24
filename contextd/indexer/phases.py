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
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from contextd.corpus_config import CorpusConfig
from contextd.indexer.hasher import FileHasher
from contextd.indexer.heading_parser import HeadingParser, ParsedSection
from contextd.inference.relate import RelationshipInferrer
from contextd.inference.summarise import Summariser
from contextd.providers.base import EmbeddingProvider
from contextd.storage._keys import primary_key_for
from contextd.storage.base import GraphStore

_log = logging.getLogger(__name__)

_T = TypeVar("_T")


@dataclass
class PhaseResult:
    name: str
    processed: int
    skipped: int


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
        texts = [f.read_text(errors="replace") for f in batch]
        all_embeddings.extend(embedder.embed(texts))

    processed = 0
    for f, vec in zip(files, all_embeddings, strict=True):
        store.upsert_node(
            "File",
            {
                "path": str(f),
                "name": f.name,
                "type": f.suffix.lstrip(".") or "unknown",
                "hash": hasher.hash(f),
                "size": f.stat().st_size,
                "corpus": corpus,
                "embedding": vec,
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
) -> PhaseResult:
    # Idempotent resume: skip files whose File node already has a summary.
    # One batch lookup against the store; set-subtracted from the input list.
    if files:
        already = {
            r["path"]
            for r in store.exec_read(
                "MATCH (f:File) WHERE f.path IN $paths AND f.summary IS NOT NULL "
                "RETURN f.path AS path",
                {"paths": [str(f) for f in files]},
            )
        }
        files = [f for f in files if str(f) not in already]

    def _worker(f: Path) -> tuple[int, int]:
        try:
            result = summariser.summarise(f.read_text(errors="replace"))
        except Exception:
            return (0, 1)
        store.exec_write(
            "MATCH (n:File {path: $path}) "
            "SET n.summary = $summary, n.key_points = $key_points, n.summary_confidence = 1.0",
            {"path": str(f), "summary": result.summary, "key_points": result.key_points},
        )
        return (1, 0)

    processed, skipped = _parallel_map(files, _worker, concurrency)
    return PhaseResult(name="summarise", processed=processed, skipped=skipped)


def phase_relate(
    files: list[Path],
    inferrer: RelationshipInferrer,
    store: GraphStore,
    entity_sampler: Callable[[GraphStore], list[str]],
    *,
    corpus: str,
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
                {"paths": [str(f) for f in files]},
            )
        }
        files = [f for f in files if str(f) not in already]

    known = entity_sampler(store)

    def _worker(f: Path) -> tuple[int, int]:
        try:
            relations = inferrer.infer(f.read_text(errors="replace"), known_entities=known)
        except Exception:
            return (0, 1)
        # Wipe-and-replace inferred edges (spec §5.5).
        # src_label="File" required by GraphStore.delete_edges (see ABC
        # docstring) — a label-less MATCH is ambiguous when endpoints
        # have non-"path" PKs.
        store.delete_edges(str(f), origin="inferred", src_label="File")
        local_skipped = 0
        for rel in relations:
            try:
                pk = _infer_key(rel.target_type)
            except ValueError:
                # Hallucinated target label — skip this edge rather than
                # creating a malformed weak-entry node or aborting the
                # whole batch.
                local_skipped += 1
                continue
            # Tag the auto-created destination with the current corpus so
            # phase_gc_sections (and any future corpus-scoped cleanup) can
            # see it. Without this, stubs MERGEd here carry only their PK
            # and become untracked orphans with corpus=NULL.
            store.upsert_node(rel.target_type, {pk: rel.target_name, "corpus": corpus})
            # src_label/dst_label required by GraphStore.upsert_edge.
            store.upsert_edge(
                str(f),
                rel.target_name,
                rel.edge_type,
                origin="inferred",
                properties={"confidence": rel.confidence, "reason": rel.reason},
                src_label="File",
                dst_label=rel.target_type,
            )
        # Mark processed so an interrupted run can resume without re-inferring.
        # Marker set only after the upsert loop completes; exception paths
        # return (0, 1) above and leave the marker unset.
        store.exec_write(
            "MATCH (f:File {path: $path}) SET f.inferred_at = datetime()",
            {"path": str(f)},
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
    parser = HeadingParser(
        min_level=corpus_cfg.corpus.heading_min_level,
        max_level=corpus_cfg.corpus.heading_max_level,
    )

    # Defence-in-depth (M10.9): non-.md files yield zero sections and would
    # leave File.summary NULL.  The caller (run_bootstrap) partitions files
    # before calling this function; this guard catches future mis-routing.
    md_files: list[Path] = []
    for f in files:
        if f.suffix != ".md":
            _log.warning(
                "phase_enumerate_sections: skipping non-markdown file %s "
                "(route through phase_enumerate instead)",
                f,
            )
        else:
            md_files.append(f)
    files = md_files

    # Collect all sections for all files first so we can batch-embed in one pass.
    parsed_by_file: list[tuple[Path, list[ParsedSection]]] = [
        (f, parser.parse(f.read_text(errors="replace"))) for f in files
    ]
    all_sections: list[tuple[Path, ParsedSection]] = [
        (f, sec) for f, secs in parsed_by_file for sec in secs
    ]
    all_bodies = [sec.body for _, sec in all_sections]

    # Batch-embed all section bodies.
    embeddings: list[list[float]] = []
    for start in range(0, len(all_bodies), batch_size):
        embeddings.extend(embedder.embed(all_bodies[start : start + batch_size]))

    # Build (file_path, section_id) → embedding lookup.
    embedding_map: dict[tuple[str, str], list[float]] = {
        (str(f), f"{f!s}#{sec.anchor}"): vec
        for (f, sec), vec in zip(all_sections, embeddings, strict=True)
    }

    processed = 0
    for f, sections in parsed_by_file:
        file_path = str(f)
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
    parser = _build_parser(corpus_cfg)
    parse_cache: dict[str, list[ParsedSection]] = {}
    current_ids: set[str] = set()
    for f in files:
        file_path = str(f)
        for sec in _parse_cached(parser, f, parse_cache):
            current_ids.add(f"{file_path}#{sec.anchor}")

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
) -> int:
    """GC stale Section nodes for a single file after incremental re-index.

    Runs HeadingParser on *path*, queries Section nodes for this file, and
    DETACH DELETEs any whose anchor is no longer produced by the parser
    (renamed or deleted headings). Returns the count of deleted sections.

    Called from run_incremental_file after phase_enumerate_sections so that
    renamed headings are cleaned up without waiting for the next full bootstrap.
    """
    if path.suffix != ".md":
        return 0
    parser = _build_parser(corpus_cfg)
    file_path = str(path)
    current_ids: set[str] = {
        f"{file_path}#{sec.anchor}" for sec in parser.parse(path.read_text(errors="replace"))
    }
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
) -> PhaseResult:
    """Summarise each Section node via LLM (spec §5.11.3).

    Reads the section body by re-parsing the source file and locating the
    section by anchor. On any exception (provider error, parse failure) the
    section is skipped and counted in skipped. Parse output is cached per
    file via ``_parse_cached`` so each file is parsed once per phase.

    Under ``concurrency > 1`` the parse cache is pre-populated serially
    before workers are dispatched; dict reads from multiple threads are
    safe, dict writes are not.
    """
    # Idempotent resume: skip Section nodes that already have a summary.
    rows = store.exec_read(
        "MATCH (s:Section {corpus: $c}) WHERE s.summary IS NULL RETURN s.id AS id, s.path AS path",
        {"c": corpus_cfg.corpus.name},
    )
    parser = _build_parser(corpus_cfg)
    parse_cache: dict[str, list[ParsedSection]] = {}
    for p in {Path(r["path"]) for r in rows}:
        _parse_cached(parser, p, parse_cache)

    def _worker(r: dict[str, str]) -> tuple[int, int]:
        path = Path(r["path"])
        sections = _parse_cached(parser, path, parse_cache)
        anchor = r["id"].split("#", 1)[1]
        sec = next((s for s in sections if s.anchor == anchor), None)
        if not sec:
            return (0, 1)
        try:
            result = summariser.summarise(sec.body)
        except Exception:
            return (0, 1)
        store.exec_write(
            "MATCH (s:Section {id: $id}) "
            "SET s.summary = $summary, s.key_points = $key_points, s.summary_confidence = 1.0",
            {
                "id": r["id"],
                "summary": result.summary,
                "key_points": result.key_points,
            },
        )
        return (1, 0)

    processed, skipped = _parallel_map(rows, _worker, concurrency)
    return PhaseResult(name="summarise_sections", processed=processed, skipped=skipped)


def phase_relate_sections(
    corpus_cfg: CorpusConfig,
    inferrer: RelationshipInferrer,
    store: GraphStore,
    entity_sampler: Callable[[GraphStore], list[str]],
    *,
    concurrency: int = 1,
) -> PhaseResult:
    """Infer typed edges from each Section node (spec §5.11.3).

    Wipe-and-replace inferred edges per section then upsert new ones.
    ``delete_edges`` and ``upsert_edge`` both supply ``src_label="Section"``
    and ``dst_label=rel.target_type`` as required by ``GraphStore``. Parse
    output is cached per file via ``_parse_cached`` so each file is parsed
    once per phase.

    Under ``concurrency > 1`` the parse cache is pre-populated serially
    before workers are dispatched (see ``phase_summarise_sections``).
    """
    # Idempotent resume: skip Sections that already carry an inferred_at
    # marker. Zero-edge sections still get marked (see worker below) so
    # they are not re-attempted on every restart.
    rows = store.exec_read(
        "MATCH (s:Section {corpus: $c}) WHERE s.inferred_at IS NULL "
        "RETURN s.id AS id, s.path AS path",
        {"c": corpus_cfg.corpus.name},
    )
    parser = _build_parser(corpus_cfg)
    parse_cache: dict[str, list[ParsedSection]] = {}
    for p in {Path(r["path"]) for r in rows}:
        _parse_cached(parser, p, parse_cache)
    known = entity_sampler(store)

    def _worker(r: dict[str, str]) -> tuple[int, int]:
        path = Path(r["path"])
        sections = _parse_cached(parser, path, parse_cache)
        anchor = r["id"].split("#", 1)[1]
        sec = next((s for s in sections if s.anchor == anchor), None)
        if not sec:
            return (0, 1)
        try:
            relations = inferrer.infer(sec.body, known_entities=known)
        except Exception:
            return (0, 1)
        # Wipe-and-replace inferred edges for this section (spec §5.5).
        store.delete_edges(r["id"], origin="inferred", src_label="Section")
        local_skipped = 0
        for rel in relations:
            try:
                pk = _infer_key(rel.target_type)
            except ValueError:
                # Hallucinated target label — skip silently (see phase_relate
                # for the same pattern in file-mode).
                local_skipped += 1
                continue
            # Tag the auto-created destination with the current corpus so
            # phase_gc_sections (and any future corpus-scoped cleanup) can
            # see it. Without this, stubs MERGEd here carry only their PK
            # and become untracked orphans with corpus=NULL.
            store.upsert_node(
                rel.target_type,
                {pk: rel.target_name, "corpus": corpus_cfg.corpus.name},
            )
            store.upsert_edge(
                r["id"],
                rel.target_name,
                rel.edge_type,
                origin="inferred",
                properties={"confidence": rel.confidence, "reason": rel.reason},
                src_label="Section",
                dst_label=rel.target_type,
            )
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
    file_path = str(path)
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


def _build_parser(corpus_cfg: CorpusConfig) -> HeadingParser:
    """Construct a HeadingParser from corpus config bounds."""
    return HeadingParser(
        min_level=corpus_cfg.corpus.heading_min_level,
        max_level=corpus_cfg.corpus.heading_max_level,
    )


def _parse_cached(
    parser: HeadingParser, path: Path, cache: dict[str, list[ParsedSection]]
) -> list[ParsedSection]:
    """Parse ``path`` once per phase, caching by absolute path string.

    Was: each of the three section phases re-parsed every file per
    section row. For a file with N sections, phase_summarise_sections
    and phase_relate_sections each did N parses. Now each phase does
    one parse per file, N-times cheaper.
    """
    key = str(path)
    cached = cache.get(key)
    if cached is not None:
        return cached
    sections = parser.parse(path.read_text(errors="replace"))
    cache[key] = sections
    return sections


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
