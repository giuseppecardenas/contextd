"""Shared scaffolding for the retrieval-chunk integration tests.

``test_pipeline_chunks.py`` and ``test_search_chunks.py`` both need the same
things: a small synthetic markdown corpus with the block shapes the
``structural`` strategy treats specially (a fenced code block, a table, a
long paragraph that must be split), a deterministic embedder so vector
legs run without a provider, fake LLM collaborators, and a bootstrap helper
that wires ``ChunkingDeps`` through ``build_chunking_deps`` exactly as the
CLI does. Underscore-prefixed so pytest never collects it as a test module.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from contextd.config import Config
from contextd.corpus_config import CorpusConfig
from contextd.indexer.chunk_deps import ChunkingDeps, build_chunking_deps
from contextd.indexer.hasher import FileHasher
from contextd.indexer.phases import PhaseResult, RelateDeps
from contextd.indexer.pipeline import BootstrapResult, RefreshScope, run_bootstrap
from contextd.inference.context import EmptyRetriever
from contextd.inference.summarise import FileSummary
from contextd.providers.base import EmbeddingProvider, UsageRecord
from contextd.storage.base import GraphStore

DIM = 1024  # matches the Chunk_embedding_idx dimension in migration _0008

# A token that appears in exactly one section *body* and in no summary, key
# point or heading: the headline regression the chunk layer exists to fix is
# that such a token used to be unreachable by ``search``.
BODY_ONLY_TOKEN = "zorblatt"
# Same idea for the second file, so per-file assertions have a handle.
FAQ_TOKEN = "quuxwing"

SUMMARY = "stub summary"
KEY_POINT = "stub point"


class HashEmbedder(EmbeddingProvider):
    """Deterministic hashed bag-of-words embedder.

    Every whitespace word hashes to one of ``DIM`` buckets; the vector is the
    L2-normalised bucket histogram. Equal texts get equal vectors, and a text
    that shares words with the query has a strictly positive cosine with it,
    so the vector leg of a hybrid search is meaningful (the query
    ``"zorblatt"`` really does land on the chunks that contain the word).
    Thread-safe by construction — the chunk phase calls it from workers.
    """

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _bucket(word: str) -> int:
        return int(hashlib.md5(word.lower().encode("utf-8")).hexdigest(), 16) % DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * DIM
            for word in text.split():
                vec[self._bucket(word)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm == 0.0:
                # Cosine indexes reject the zero vector; pin empty texts to a
                # fixed direction instead.
                vec[0] = 1.0
                norm = 1.0
            out.append([v / norm for v in vec])
        return out

    def last_usage(self) -> UsageRecord | None:
        return None

    @property
    def dimensions(self) -> int:
        return DIM


def _long_paragraph() -> str:
    # One sentence per line (hard-wrapped prose, no blank lines): the
    # paragraph is far over the ``fine`` cap, so the recursive fallback splits
    # it on line breaks and every chunk is a run of whole source lines — which
    # is what makes ``start_line`` / ``end_line`` evidence checkable.
    sentences: list[str] = []
    for i in range(60):
        if i == 30:
            sentences.append(
                f"Step {i} enables the {BODY_ONLY_TOKEN} flag on the rollout controller."
            )
        else:
            sentences.append(
                f"Step {i} of the rollout applies configuration item {i} to the deployment target."
            )
    return "\n".join(sentences)


def _fence() -> str:
    # ~245 word-tokens: under the default ``fine`` cap of 256 so it stays one
    # fence, but too big to pack with the prose around it, so it is emitted
    # as a standalone ``code`` chunk rather than folded into a ``mixed`` one.
    return "```toml\n" + "".join(f"key_{i} = {i}\n" for i in range(62)) + "```\n"


def _table() -> str:
    header = "| name | type | default |\n|------|------|---------|\n"
    return header + "".join(f"| option-{i} | string | value-{i} |\n" for i in range(40))


GUIDE_MD = f"""# Deployment Guide

This guide covers the rollout of the service.

## Overview

{_long_paragraph()}

## Configuration

The service reads its settings from a TOML file.

{_fence()}
Restart the service after editing the file so the new values take effect, and check the
startup log for any key the parser rejected before routing traffic to it.

## Reference Table

{_table()}
## Notes

Keep this section short.
"""

FAQ_MD = f"""# FAQ

Questions about the service.

## Installing

Install the package with the standard tooling. The {FAQ_TOKEN} installer handles dependencies.

## Upgrading

Upgrades are applied in place.
"""

# Sections the heading parser yields for the two files at the default
# heading levels: each file's preamble (titled after its H1) plus its H2s.
GUIDE_SECTIONS = ("Deployment Guide", "Overview", "Configuration", "Reference Table", "Notes")
FAQ_SECTIONS = ("FAQ", "Installing", "Upgrading")


def write_corpus(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "guide.md").write_text(GUIDE_MD, encoding="utf-8")
    (root / "faq.md").write_text(FAQ_MD, encoding="utf-8")


def corpus_config(
    root: Path,
    name: str = "chunks",
    *,
    granularity: str = "section",
    include: tuple[str, ...] = ("*.md",),
    profiles: list[dict[str, Any]] | None = None,
) -> CorpusConfig:
    """Corpus config with ``[chunking] tokenizer = "words"`` forced.

    The word tokenizer keeps the test offline (``auto`` would try to fetch
    the Voyage tokenizer from the Hub) and deterministic. ``profiles`` of
    ``None`` keeps the two shipped defaults (``fine`` / ``coarse``).
    """
    chunking: dict[str, Any] = {"tokenizer": "words"}
    if profiles is not None:
        chunking["profiles"] = profiles
    return CorpusConfig.model_validate(
        {
            "corpus": {
                "name": name,
                "root": str(root),
                "include": list(include),
                "granularity": granularity,
            },
            "chunking": chunking,
        }
    )


def fake_summariser(summary: str = SUMMARY) -> MagicMock:
    s = MagicMock()
    s.summarise.return_value = FileSummary(
        summary=summary, key_points=[KEY_POINT], entities_mentioned=[]
    )
    s.roll_up.return_value = f"rolled {summary}"
    return s


def relate_deps(inferrer: MagicMock | None = None) -> RelateDeps:
    if inferrer is None:
        inferrer = MagicMock()
        inferrer.infer.return_value = []
    return RelateDeps(inferrer=inferrer, retriever=EmptyRetriever())


def chunking_deps(corpus_cfg: CorpusConfig, embedder: EmbeddingProvider) -> ChunkingDeps:
    """Build the deps the way the CLI does: from a real ``Config`` plus the corpus config."""
    deps = build_chunking_deps(
        Config(), corpus_cfg, embedder=embedder, inference=None, renderer=None
    )
    assert deps is not None, "chunking is enabled by default"
    return deps


def bootstrap(
    backend: GraphStore,
    corpus_cfg: CorpusConfig,
    *,
    embedder: HashEmbedder | None = None,
    summariser: MagicMock | None = None,
    relate: RelateDeps | None = None,
    refresh: RefreshScope | None = None,
    chunking: ChunkingDeps | None = None,
) -> tuple[BootstrapResult, ChunkingDeps]:
    """Run ``run_bootstrap`` with chunking wired in; return the result and the deps used."""
    emb = embedder if embedder is not None else HashEmbedder()
    deps = chunking if chunking is not None else chunking_deps(corpus_cfg, emb)
    result = run_bootstrap(
        corpus=corpus_cfg,
        store=backend,
        embedder=emb,
        summariser=summariser if summariser is not None else fake_summariser(),
        relate=relate if relate is not None else relate_deps(),
        hasher=FileHasher(),
        refresh=refresh,
        chunking=deps,
    )
    return result, deps


def phase(result: BootstrapResult, name: str) -> PhaseResult:
    matches = [p for p in result.phases if p.name == name]
    assert len(matches) == 1, f"expected exactly one {name!r} phase, got {matches}"
    return matches[0]


def chunk_rows(backend: GraphStore, corpus: str) -> list[dict[str, Any]]:
    """Every Chunk of ``corpus`` (embedding omitted), ordered by parent, profile, ordinal."""
    return backend.exec_read(
        "MATCH (c:Chunk {corpus: $c}) "
        "RETURN c.id AS id, c.path AS path, c.parent_id AS parent_id, "
        "c.parent_label AS parent_label, c.profile AS profile, c.ordinal AS ordinal, "
        "c.kind AS kind, c.part AS part, c.hash AS hash, c.token_count AS token_count, "
        "c.start_line AS start_line, c.end_line AS end_line, c.text AS text "
        "ORDER BY c.parent_id, c.profile, c.ordinal",
        {"c": corpus},
    )


def section_fingerprints(backend: GraphStore, corpus: str) -> dict[str, str | None]:
    return {
        r["id"]: r["fp"]
        for r in backend.exec_read(
            "MATCH (s:Section {corpus: $c}) RETURN s.id AS id, s.chunk_fingerprint AS fp",
            {"c": corpus},
        )
    }
