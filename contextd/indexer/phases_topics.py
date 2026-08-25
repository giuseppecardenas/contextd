"""``phase_cluster_topics``: RAPTOR-style cross-document topic tree.

Layer 0 clusters the corpus's Section (or File) embeddings; each cluster
becomes a ``Topic`` node whose title/summary the LLM writes from the members'
summaries and whose embedding is that summary's vector. Layer ``n + 1``
clusters the layer-``n`` topic embeddings, up to ``max_layers`` or until a
layer yields a single cluster. Membership is a ``BELONGS_TO
{origin:"structural", probability}`` edge from the member (Section, File
or lower-layer Topic) to the topic.

Gated on ``Corpus.topic_input_fingerprint`` — a hash of the member ids and
their ``summary_input_hash`` / summaries — so an unchanged corpus costs
nothing on re-run; the daemon reclusters when ``Corpus.topics_dirty`` is set
by an incremental pass (see ``contextd.daemon``).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from contextd.corpus_config import CorpusConfig
from contextd.indexer.chunk_deps import ChunkingDeps
from contextd.indexer.phases import PhaseResult
from contextd.storage.base import GraphStore
from contextd.topics.cluster import Cluster, cluster_vectors, split_oversize
from contextd.topics.summarise import summarise_topic

_log = logging.getLogger(__name__)


@dataclass
class _Member:
    id: str
    label: str
    summary: str
    embedding: list[float]
    tokens: int


def _load_members(store: GraphStore, corpus_cfg: CorpusConfig, deps: ChunkingDeps) -> list[_Member]:
    corpus = corpus_cfg.corpus.name
    if corpus_cfg.topics.source == "section":
        rows = store.exec_read(
            "MATCH (s:Section {corpus: $c}) WHERE s.embedding IS NOT NULL AND s.summary IS NOT NULL "
            "RETURN s.id AS id, 'Section' AS label, s.summary AS summary, s.embedding AS embedding "
            "ORDER BY s.id",
            {"c": corpus},
        )
    else:
        rows = store.exec_read(
            "MATCH (f:File {corpus: $c}) WHERE f.embedding IS NOT NULL AND f.summary IS NOT NULL "
            "RETURN f.path AS id, 'File' AS label, f.summary AS summary, f.embedding AS embedding "
            "ORDER BY f.path",
            {"c": corpus},
        )
    members: list[_Member] = []
    for r in rows:
        emb = r.get("embedding")
        if not isinstance(emb, list) or not emb:
            continue
        summary = str(r.get("summary") or "")
        members.append(
            _Member(
                id=str(r["id"]),
                label=str(r["label"]),
                summary=summary,
                embedding=[float(x) for x in emb],
                tokens=deps.tokenizer.count(summary),
            )
        )
    return members


def input_fingerprint(members: list[_Member], corpus_cfg: CorpusConfig) -> str:
    h = hashlib.sha256()
    h.update(corpus_cfg.topics.model_dump_json().encode("utf-8"))
    for m in members:
        h.update(m.id.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.md5(m.summary.encode("utf-8")).digest())
    return h.hexdigest()


def _stored_fingerprint(store: GraphStore, corpus: str) -> tuple[str | None, bool]:
    rows = store.exec_read(
        "MATCH (n:Corpus {name: $c}) RETURN n.topic_input_fingerprint AS fp, n.topics_dirty AS dirty",
        {"c": corpus},
    )
    if not rows:
        return None, False
    return rows[0].get("fp"), bool(rows[0].get("dirty"))


def _cluster_layer(members: list[_Member], corpus_cfg: CorpusConfig, seed: int) -> list[Cluster]:
    t = corpus_cfg.topics
    vectors = [m.embedding for m in members]
    clusters = cluster_vectors(
        vectors,
        min_members=t.min_members,
        soft_threshold=t.soft_threshold,
        pca_dims=t.pca_dims,
        seed=seed,
    )
    return split_oversize(
        vectors,
        [m.tokens for m in members],
        clusters,
        max_cluster_tokens=t.max_cluster_tokens,
        min_members=t.min_members,
        soft_threshold=t.soft_threshold,
        pca_dims=t.pca_dims,
        seed=seed,
    )


def phase_cluster_topics(
    corpus_cfg: CorpusConfig,
    deps: ChunkingDeps,
    store: GraphStore,
    *,
    max_words: int = 100,
    force: bool = False,
) -> PhaseResult:
    """Build (or rebuild) the corpus's topic tree; returns topics written."""
    corpus = corpus_cfg.corpus.name
    if not corpus_cfg.topics.enabled:
        return PhaseResult(name="cluster_topics", processed=0, skipped=0)
    if deps.inference is None or deps.renderer is None or deps.embedder is None:
        _log.warning("topics: inference/embedding providers unavailable; skipping")
        return PhaseResult(name="cluster_topics", processed=0, skipped=1)
    members = _load_members(store, corpus_cfg, deps)
    if len(members) < corpus_cfg.topics.min_members:
        _log.info("topics: corpus %s has %d members; nothing to cluster", corpus, len(members))
        return PhaseResult(name="cluster_topics", processed=0, skipped=len(members))
    fingerprint = input_fingerprint(members, corpus_cfg)
    stored, dirty = _stored_fingerprint(store, corpus)
    if not force and not dirty and stored == fingerprint:
        return PhaseResult(name="cluster_topics", processed=0, skipped=len(members))

    store.delete_nodes("Topic", where={"corpus": corpus})
    now = dt.datetime.now(dt.UTC)
    written = 0
    layer_members = members
    for layer in range(corpus_cfg.topics.max_layers):
        clusters = _cluster_layer(layer_members, corpus_cfg, corpus_cfg.topics.seed + layer)
        if layer > 0 and len(clusters) <= 1:
            break
        topic_rows: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        next_members: list[_Member] = []
        for n, cluster in enumerate(clusters):
            ordered = sorted(
                zip(cluster.members, cluster.probabilities, strict=True), key=lambda t: -t[1]
            )
            summaries = [layer_members[i].summary for i, _ in ordered if layer_members[i].summary]
            text = summarise_topic(
                deps.inference, deps.renderer, summaries, corpus=corpus, max_words=max_words
            )
            if text is None:
                continue
            try:
                [vec] = deps.embedder.embed([text.summary])
            except Exception as exc:
                _log.warning("topics: embed failed for layer %d topic %d: %s", layer, n, exc)
                continue
            topic_id = f"{corpus}/topic/{layer}/{n}"
            topic_rows.append(
                {
                    "id": topic_id,
                    "corpus": corpus,
                    "layer": layer,
                    "title": text.title,
                    "summary": text.summary,
                    "embedding": vec,
                    "member_count": len(cluster.members),
                    "updated": now,
                }
            )
            for i, prob in ordered:
                m = layer_members[i]
                edges.append({"src": m.id, "label": m.label, "dst": topic_id, "p": float(prob)})
            next_members.append(
                _Member(
                    id=topic_id,
                    label="Topic",
                    summary=text.summary,
                    embedding=vec,
                    tokens=deps.tokenizer.count(text.summary),
                )
            )
        if not topic_rows:
            break
        store.upsert_nodes("Topic", topic_rows)
        for label in {e["label"] for e in edges}:
            pk = "path" if label == "File" else "id"
            store.exec_write(
                f"UNWIND $edges AS e MATCH (m:{label} {{{pk}: e.src}}), (t:Topic {{id: e.dst}}) "
                "MERGE (m)-[r:BELONGS_TO]->(t) SET r.origin = 'structural', r.probability = e.p",
                {"edges": [e for e in edges if e["label"] == label]},
            )
        written += len(topic_rows)
        if len(next_members) < 2 * corpus_cfg.topics.min_members:
            break
        layer_members = next_members

    store.exec_write(
        "MATCH (n:Corpus {name: $c}) SET n.topic_input_fingerprint = $fp, "
        "n.topics_dirty = false, n.topic_count = $n",
        {"c": corpus, "fp": fingerprint, "n": written},
    )
    return PhaseResult(name="cluster_topics", processed=written, skipped=0)


def topics_dirty(store: GraphStore, corpus: str) -> bool:
    _, dirty = _stored_fingerprint(store, corpus)
    return dirty
