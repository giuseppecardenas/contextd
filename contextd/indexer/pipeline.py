"""Coordinates bootstrap + incremental indexing flow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from contextd.corpus_config import CorpusConfig
from contextd.indexer import phases
from contextd.indexer.hasher import FileHasher
from contextd.inference.relate import RelationshipInferrer
from contextd.inference.summarise import Summariser
from contextd.providers.base import EmbeddingProvider
from contextd.storage.base import GraphStore


@dataclass
class BootstrapResult:
    phases: list[phases.PhaseResult]


def enumerate_corpus_files(corpus: CorpusConfig) -> list[Path]:
    root = Path(corpus.corpus.root).expanduser()
    hits: list[Path] = []
    for pattern in corpus.corpus.include:
        hits.extend(root.glob(pattern))
    excl = {root / e for e in corpus.corpus.exclude}
    return [p for p in hits if p.is_file() and p not in excl]


def run_bootstrap(
    corpus: CorpusConfig,
    store: GraphStore,
    embedder: EmbeddingProvider,
    summariser: Summariser,
    inferrer: RelationshipInferrer,
    hasher: FileHasher,
    entity_sampler: Callable[[GraphStore], list[str]],
) -> BootstrapResult:
    files = enumerate_corpus_files(corpus)
    results: list[phases.PhaseResult] = []
    # Spec-delta (b): phase_enumerate now accepts embedder so that embedding
    # vectors are included in the initial upsert_node call (Kuzu requires
    # embedding at CREATE time; File.embedding is IMMUTABLE_AFTER_CREATE).
    results.append(phases.phase_enumerate(files, corpus.corpus.name, hasher, store, embedder))
    # phase_embed is an accounting-only pass (embedding already done in enumerate).
    results.append(phases.phase_embed(files, embedder, store))
    results.append(phases.phase_summarise(files, summariser, store))
    results.append(phases.phase_relate(files, inferrer, store, entity_sampler))
    results.append(phases.phase_close(corpus.corpus.name, store, results))
    return BootstrapResult(phases=results)
