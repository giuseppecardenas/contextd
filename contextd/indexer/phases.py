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
            # Upsert target node (structural or weak entry) then edge.
            store.upsert_node(rel.target_type, {_infer_key(rel.target_type): rel.target_name})
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


def phase_embed_sections(
    corpus_cfg: CorpusConfig,
    embedder: EmbeddingProvider,
    store: GraphStore,
) -> PhaseResult:
    """M9.2 placeholder — section embedding is done at CREATE time in phase_enumerate_sections.

    TODO(M9.2): implement if a separate embed-update pass is needed for incremental mode.
    """
    return PhaseResult(name="embed_sections", processed=0, skipped=0)


def phase_summarise_sections(
    corpus_cfg: CorpusConfig,
    summariser: Summariser,
    store: GraphStore,
) -> PhaseResult:
    """M9.2 placeholder — summarise each Section node via LLM.

    TODO(M9.2): implement per-section summarisation (mirrors phase_summarise for files).
    """
    return PhaseResult(name="summarise_sections", processed=0, skipped=0)


def phase_relate_sections(
    corpus_cfg: CorpusConfig,
    inferrer: RelationshipInferrer,
    store: GraphStore,
    entity_sampler: Callable[[GraphStore], list[str]],
) -> PhaseResult:
    """M9.2 placeholder — infer typed edges between Section nodes.

    TODO(M9.2): implement per-section relationship inference (mirrors phase_relate for files).
    """
    return PhaseResult(name="relate_sections", processed=0, skipped=0)


def phase_derive_file_level(
    corpus_cfg: CorpusConfig,
    store: GraphStore,
) -> PhaseResult:
    """M9.2 placeholder — roll up section summaries/embeddings to parent File node.

    TODO(M9.2): implement aggregation from Section → File (spec §5.11.3).
    """
    return PhaseResult(name="derive_file_level", processed=0, skipped=0)


def _infer_key(target_type: str) -> str:
    """Return the primary-key property name for target_type.

    Deferred item: should route through primary_key_for() from
    contextd.storage._keys instead of this hardcoded subset. _keys.py knows
    14 labels; this map covers 4. Any new target label with a different PK
    (e.g., Risk.description) will silently return "name". Address during M8/M9.
    """
    return {"File": "path", "Section": "id", "Artifact": "id", "Ticket": "id"}.get(
        target_type, "name"
    )
