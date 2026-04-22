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
) -> BootstrapResult:
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
                files, inferrer, store, entity_sampler, concurrency=inference_concurrency
            )
        )
        results.append(phases.phase_close(corpus.corpus.name, store, results))
    return BootstrapResult(phases=results)
