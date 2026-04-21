"""Five-phase bootstrap pipeline per spec §5.9 Step 5.

Phase 5a: enumeration       (walk corpus, hash files, create File nodes with embeddings)
Phase 5b: embedding         (accounting phase — embeddings were created in 5a; returns count)
Phase 5c: summarisation     (Gemini per-file → File.summary + key_points)
Phase 5d: relationship inf. (Gemini per-file → typed edges, wipe-and-replace inferred)
Phase 5e: corpus closure    (write Corpus singleton stats)

Spec-delta note (b): The plan's phase_embed used exec_write("... SET n.embedding = $vec",
...) which fails on Kuzu because File.embedding is in
IMMUTABLE_AFTER_CREATE_BY_LABEL — Kuzu rejects SET on vector-indexed columns after
node creation. Resolution: embedding is computed in batch during phase_enumerate and
included in the initial upsert_node call (so embedding is set at CREATE time).
phase_embed is retained as a named accounting phase that reports the count without
re-issuing writes, preserving the 5-phase contract and the integration test assertion
result.phases[1].processed == 2. This is a structural deviation from the plan text;
logged as spec-delta (b).

Spec-delta note (M9.1-A): phase_enumerate_sections accepts embedder and pre-computes
Section embeddings in batch at CREATE time. Section.embedding is IMMUTABLE_AFTER_CREATE
on Kuzu; SET after creation would fail. Mirrors the M5.4 pattern for File.embedding.

Spec-delta note (M9.1-B): upsert_edge calls in phase_enumerate_sections supply
src_label/dst_label kwargs. Kuzu requires these for schema-first REL tables:
  CONTAINS (FROM File TO Section), PARENT_OF (FROM Section TO Section),
  NEXT_SIBLING (FROM Section TO Section).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from contextd.corpus_config import CorpusConfig
from contextd.indexer.hasher import FileHasher
from contextd.indexer.heading_parser import HeadingParser, ParsedSection
from contextd.inference.relate import RelationshipInferrer
from contextd.inference.summarise import Summariser
from contextd.providers.base import EmbeddingProvider
from contextd.storage._keys import primary_key_for
from contextd.storage.base import GraphStore


@dataclass
class PhaseResult:
    name: str
    processed: int
    skipped: int


def phase_enumerate(
    files: list[Path],
    corpus: str,
    hasher: FileHasher,
    store: GraphStore,
    embedder: EmbeddingProvider,
    batch_size: int = 128,
) -> PhaseResult:
    """Create File nodes with embeddings included at creation time.

    Spec-delta (b): embedder is accepted here so that embedding vectors are
    passed to upsert_node at CREATE time. Kuzu rejects SET on vector-indexed
    columns (File.embedding) after node creation; including the embedding in
    the initial properties dict satisfies the constraint. The phase_embed step
    below is a count-only accounting pass.
    """
    # Batch-compute embeddings for all files upfront so we can include them
    # in the initial upsert_node call (required by Kuzu's IMMUTABLE_AFTER_CREATE).
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

    Spec-delta (b): phase_enumerate embeds at node-CREATE time because
    Kuzu's File.embedding is IMMUTABLE_AFTER_CREATE. This phase reports
    the count of files that were embedded, preserving the 5-phase contract
    and the integration test's phases[1].processed assertion.
    """
    return PhaseResult(name="embed", processed=len(files), skipped=0)


def phase_summarise(
    files: list[Path],
    summariser: Summariser,
    store: GraphStore,
) -> PhaseResult:
    processed, skipped = 0, 0
    for f in files:
        try:
            result = summariser.summarise(f.read_text(errors="replace"))
        except Exception:
            skipped += 1
            continue
        store.exec_write(
            "MATCH (n:File {path: $path}) "
            "SET n.summary = $summary, n.key_points = $key_points, n.summary_confidence = 1.0",
            {"path": str(f), "summary": result.summary, "key_points": result.key_points},
        )
        processed += 1
    return PhaseResult(name="summarise", processed=processed, skipped=skipped)


def phase_relate(
    files: list[Path],
    inferrer: RelationshipInferrer,
    store: GraphStore,
    entity_sampler: Callable[[GraphStore], list[str]],
) -> PhaseResult:
    processed, skipped = 0, 0
    known = entity_sampler(store)
    for f in files:
        try:
            relations = inferrer.infer(f.read_text(errors="replace"), known_entities=known)
        except Exception:
            skipped += 1
            continue
        # Wipe-and-replace inferred edges (spec §5.5).
        # Spec-delta (c): src_label="File" added — Kuzu requires src_label on
        # delete_edges; without it KuzuBackend raises ValueError.
        store.delete_edges(str(f), origin="inferred", src_label="File")
        for rel in relations:
            try:
                pk = _infer_key(rel.target_type)
            except ValueError:
                # Hallucinated target label — skip this edge rather than
                # creating a malformed weak-entry node or aborting the
                # whole batch.
                skipped += 1
                continue
            store.upsert_node(rel.target_type, {pk: rel.target_name})
            # Spec-delta (c): src_label="File", dst_label=rel.target_type added —
            # Kuzu requires both labels on upsert_edge for schema-first REL tables.
            store.upsert_edge(
                str(f),
                rel.target_name,
                rel.edge_type,
                origin="inferred",
                properties={"confidence": rel.confidence, "reason": rel.reason},
                src_label="File",
                dst_label=rel.target_type,
            )
        processed += 1
    return PhaseResult(name="relate", processed=processed, skipped=skipped)


def phase_close(
    corpus: str,
    store: GraphStore,
    results: list[PhaseResult],
) -> PhaseResult:
    # Spec-delta (d): replaced "__now__" placeholder with datetime.now(timezone.utc).
    # Kuzu's TIMESTAMP column rejects the string "__now__" — no backend code performed
    # the substitution. Using a real UTC datetime at the call site is the correct fix.
    # Spec-delta (f-extended): node_count / edge_count not persisted on the
    # Corpus node because Kuzu's Corpus schema does not declare those columns.
    # Adding them requires a migration; deferred until the MCP server needs them.
    store.upsert_node(
        "Corpus",
        {
            "name": corpus,
            "registered_at": dt.datetime.now(dt.UTC),
        },
    )
    return PhaseResult(name="close", processed=1, skipped=0)


def phase_enumerate_sections(
    files: list[Path],
    corpus_cfg: CorpusConfig,
    store: GraphStore,
    embedder: EmbeddingProvider,
    batch_size: int = 128,
) -> PhaseResult:
    """Section-granular enumeration — emits Section nodes + structural edges.

    Spec-delta (M9.1-A): embedder is accepted here so that Section.embedding
    is included in upsert_node at CREATE time. Kuzu rejects SET on
    IMMUTABLE_AFTER_CREATE columns (Section.embedding) after node creation.
    Batch-embed all section bodies first, then upsert with embedding attached.

    Spec-delta (M9.1-B): upsert_edge calls supply src_label/dst_label kwargs
    because Kuzu requires them for schema-first REL tables.

    Spec-delta (M9.1-E): File.hash is set to "__pending__" as a sentinel
    placeholder. The hasher is not threaded through the section pipeline in
    this task. Incremental re-index cannot compare hashes reliably in section
    mode until hasher is wired here.
    TODO(M9-followup or M11): compute real MD5 via FileHasher instead of
    "__pending__" so incremental re-index works in section granularity mode.
    """
    parser = HeadingParser(
        min_level=corpus_cfg.corpus.heading_min_level,
        max_level=corpus_cfg.corpus.heading_max_level,
    )

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
        # Upsert the parent File node (hash sentinel — see TODO above).
        store.upsert_node(
            "File",
            {
                "path": file_path,
                "name": f.name,
                "type": f.suffix.lstrip(".") or "unknown",
                "hash": "__pending__",
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
            # Delta B: labels required for Kuzu schema-first REL tables.
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


def phase_embed_sections(corpus_cfg: CorpusConfig, store: GraphStore) -> PhaseResult:
    """Accounting phase: Section embeddings are written at CREATE time in
    phase_enumerate_sections (spec-delta #21 — Kuzu's Section.embedding is
    IMMUTABLE_AFTER_CREATE). This phase counts rows and returns.

    Signature shrunk in follow-up to SD #81: the ``embedder`` and
    ``batch_size`` params that mirrored the plan's pre-delta shape were
    unused — dropping them to match the M5.4 cleanup pattern from
    `1591925` (file-mode ``phase_embed(files)`` also lost dead params).

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
) -> PhaseResult:
    """Summarise each Section node via LLM (spec §5.11.3).

    Reads the section body by re-parsing the source file and locating the
    section by anchor. On any exception (provider error, parse failure) the
    section is skipped and counted in skipped.

    Spec-delta (M9.2-F): each section requires a full file re-parse; for an
    N-section file that is N parses per phase. Deferred: cache
    ParsedSection-by-anchor in-memory or store body as a graph property.
    """
    rows = store.exec_read(
        "MATCH (s:Section {corpus: $c}) RETURN s.id AS id, s.path AS path",
        {"c": corpus_cfg.corpus.name},
    )
    parser = _build_parser(corpus_cfg)
    processed, skipped = 0, 0
    for r in rows:
        path = Path(r["path"])
        sections = parser.parse(path.read_text(errors="replace"))
        anchor = r["id"].split("#", 1)[1]
        sec = next((s for s in sections if s.anchor == anchor), None)
        if not sec:
            skipped += 1
            continue
        try:
            result = summariser.summarise(sec.body)
        except Exception:
            skipped += 1
            continue
        store.exec_write(
            "MATCH (s:Section {id: $id}) "
            "SET s.summary = $summary, s.key_points = $key_points, s.summary_confidence = 1.0",
            {
                "id": r["id"],
                "summary": result.summary,
                "key_points": result.key_points,
            },
        )
        processed += 1
    return PhaseResult(name="summarise_sections", processed=processed, skipped=skipped)


def phase_relate_sections(
    corpus_cfg: CorpusConfig,
    inferrer: RelationshipInferrer,
    store: GraphStore,
    entity_sampler: Callable[[GraphStore], list[str]],
) -> PhaseResult:
    """Infer typed edges from each Section node (spec §5.11.3).

    Wipe-and-replace inferred edges per section then upsert new ones.

    Spec-delta (M9.2-B): delete_edges and upsert_edge both supply
    src_label="Section" and dst_label=rel.target_type. Kuzu requires
    explicit src/dst labels on REL-table operations; without them
    KuzuBackend raises ValueError.

    Spec-delta (M9.2-F): each section re-parses its source file (see
    phase_summarise_sections note).

    Spec-delta (M9.2-G): inferred edges from Section to arbitrary target
    types may not match Kuzu's REL-table declarations (e.g., only
    REFERENCES(FROM Section TO Section) exists). If Kuzu raises on an
    unsupported Section→X edge type, the caller will surface the error.
    The integration test guards against this by using an inferrer that
    returns no relations.
    """
    rows = store.exec_read(
        "MATCH (s:Section {corpus: $c}) RETURN s.id AS id, s.path AS path",
        {"c": corpus_cfg.corpus.name},
    )
    parser = _build_parser(corpus_cfg)
    known = entity_sampler(store)
    processed, skipped = 0, 0
    for r in rows:
        path = Path(r["path"])
        sections = parser.parse(path.read_text(errors="replace"))
        anchor = r["id"].split("#", 1)[1]
        sec = next((s for s in sections if s.anchor == anchor), None)
        if not sec:
            skipped += 1
            continue
        try:
            relations = inferrer.infer(sec.body, known_entities=known)
        except Exception:
            skipped += 1
            continue
        # Wipe-and-replace inferred edges for this section (spec §5.5).
        store.delete_edges(r["id"], origin="inferred", src_label="Section")
        for rel in relations:
            try:
                pk = _infer_key(rel.target_type)
            except ValueError:
                # Hallucinated target label — skip silently (see phase_relate
                # for the same pattern in file-mode).
                skipped += 1
                continue
            store.upsert_node(rel.target_type, {pk: rel.target_name})
            store.upsert_edge(
                r["id"],
                rel.target_name,
                rel.edge_type,
                origin="inferred",
                properties={"confidence": rel.confidence, "reason": rel.reason},
                src_label="Section",
                dst_label=rel.target_type,
            )
        processed += 1
    return PhaseResult(name="relate_sections", processed=processed, skipped=skipped)


def phase_derive_file_level(
    corpus_cfg: CorpusConfig,
    store: GraphStore,
) -> PhaseResult:
    """Derive File.summary from child section summaries (spec §5.11.3).

    Spec-delta (M9.2-C): File.embedding is NOT derived in section mode
    because Kuzu's File.embedding is IMMUTABLE_AFTER_CREATE (see
    contextd/storage/_keys.py). SET f.embedding after node creation is
    rejected by Kuzu. Centroid logic is retained via _centroid() for when a
    migration to make embedding mutable (or a DETACH-DELETE + re-CREATE
    pattern) lands. Known limitation: File.embedding is NULL in
    section-mode corpora.
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
