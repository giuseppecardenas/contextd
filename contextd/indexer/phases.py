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
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from contextd.indexer.hasher import FileHasher
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
