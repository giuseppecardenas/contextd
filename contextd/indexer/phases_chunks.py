"""Retrieval-chunk phases: ``phase_chunk_units`` and ``phase_gc_chunks``.

Chunks are derived data beneath every ``Section`` (section corpora) or
``File`` (file corpora, and non-markdown files in section corpora). A parent
is (re)chunked when its stored ``chunk_fingerprint`` differs from
``unit_fingerprint(config_fp, parent.hash)`` — one string compare that
covers resume-after-crash, incremental re-index, the daemon sweep and
"config changed" alike (plan D4).

Per parent the work is atomic in effect: delete its old chunks, upsert the
new rows in one batch, write the structural ``CONTAINS`` / ``NEXT_SIBLING``
edges, and only then stamp the fingerprint. A crash in between leaves the
parent un-stamped, so the next pass redoes it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from contextd._paths import canonical_path
from contextd.chunking.augment import apply_keywords, static_keywords
from contextd.chunking.fingerprint import unit_fingerprint
from contextd.chunking.llm import contextualise, generate_questions
from contextd.chunking.model import Chunk, ChunkRequest
from contextd.chunking.prefix import apply_prefix, breadcrumb_text, static_prefix
from contextd.corpus_config import CorpusConfig
from contextd.indexer.chunk_deps import ChunkingDeps
from contextd.indexer.phases import PhaseResult, _parallel_map, _rel_path
from contextd.indexer.units import ParseCache
from contextd.storage.base import GraphStore

_log = logging.getLogger(__name__)


@dataclass
class _Parent:
    label: str  # "Section" | "File"
    id: str  # Section.id or File.path
    path: str  # canonical file path
    content_hash: str
    stored_fp: str | None
    summary: str | None
    key_points: list[str]
    entities_mentioned: list[str]
    anchor: str | None = None


_SECTION_PARENTS = (
    "MATCH (s:Section {corpus: $c}) WHERE s.path IS NOT NULL "
    "RETURN s.id AS id, s.path AS path, s.anchor AS anchor, s.hash AS hash, "
    "s.chunk_fingerprint AS fp, s.summary AS summary, s.key_points AS key_points, "
    "s.entities_mentioned AS entities_mentioned"
)
_FILE_PARENTS = (
    "MATCH (f:File {corpus: $c}) WHERE NOT (f)-[:CONTAINS]->(:Section) "
    "RETURN f.path AS path, f.hash AS hash, f.chunk_fingerprint AS fp, "
    "f.summary AS summary, f.key_points AS key_points, "
    "f.entities_mentioned AS entities_mentioned"
)


def _str_list(raw: object) -> list[str]:
    return [str(x) for x in raw] if isinstance(raw, list) else []


def _load_parents(store: GraphStore, corpus: str, paths: set[str] | None) -> list[_Parent]:
    parents: list[_Parent] = []
    for r in store.exec_read(_SECTION_PARENTS, {"c": corpus}):
        if paths is not None and r["path"] not in paths:
            continue
        parents.append(
            _Parent(
                "Section",
                r["id"],
                r["path"],
                str(r.get("hash") or ""),
                r.get("fp"),
                r.get("summary"),
                _str_list(r.get("key_points")),
                _str_list(r.get("entities_mentioned")),
                anchor=r.get("anchor"),
            )
        )
    for r in store.exec_read(_FILE_PARENTS, {"c": corpus}):
        if paths is not None and r["path"] not in paths:
            continue
        parents.append(
            _Parent(
                "File",
                r["path"],
                r["path"],
                str(r.get("hash") or ""),
                r.get("fp"),
                r.get("summary"),
                _str_list(r.get("key_points")),
                _str_list(r.get("entities_mentioned")),
            )
        )
    return parents


def chunk_id(parent_id: str, profile: str, ordinal: int) -> str:
    return f"{parent_id}~{profile}~{ordinal}"


def _chunk_hash(profile_fp: str, chunk: Chunk) -> str:
    return hashlib.md5(f"{profile_fp}:{chunk.prefix}\n{chunk.text}".encode()).hexdigest()


def _request_for(
    parent: _Parent, corpus_cfg: CorpusConfig, cache: ParseCache, profile_name: str
) -> tuple[ChunkRequest, tuple[str, ...], str] | None:
    """Build the parent's ``ChunkRequest`` plus its breadcrumb and rel path."""
    path = Path(parent.path)
    root = Path(corpus_cfg.corpus.root)
    rel = _rel_path(path, root)
    suffix = path.suffix
    profile = next(p for p in corpus_cfg.chunking.profiles_for(suffix) if p.name == profile_name)
    if parent.label == "Section":
        try:
            parsed = cache.get(path)
        except OSError:
            return None
        assert parent.anchor is not None
        sec = parsed.by_anchor(parent.anchor)
        if sec is None:
            return None
        # Document title (the preamble's title, usually the H1) leads the
        # breadcrumb; the H1 sits below heading_min_level so it never appears
        # in the parent chain itself.
        doc_title = next((s.title for s in parsed.sections if s.is_preamble), None)
        head = (doc_title,) if doc_title and not sec.is_preamble else ()
        breadcrumb = (*head, *parsed.parent_chain(sec.anchor), sec.title)
        req = ChunkRequest(
            text=sec.body,
            profile=profile,
            blocks=corpus_cfg.chunking.blocks,
            base_line=sec.start_line,
            breadcrumb=breadcrumb,
            suffix=suffix,
        )
        return req, breadcrumb, rel
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    req = ChunkRequest(
        text=text,
        profile=profile,
        blocks=corpus_cfg.chunking.blocks,
        base_line=0,
        breadcrumb=(),
        suffix=suffix,
    )
    return req, (), rel


def _rows_for(
    parent: _Parent, profile_name: str, strategy: str, chunks: list[Chunk], corpus: str, now: Any
) -> list[dict[str, Any]]:
    return [
        {
            "id": chunk_id(parent.id, profile_name, c.ordinal),
            "corpus": corpus,
            "path": parent.path,
            "parent_id": parent.id,
            "parent_label": parent.label,
            "profile": profile_name,
            "strategy": strategy,
            "ordinal": c.ordinal,
            "kind": c.kind,
            "part": c.part,
            "text": c.text,
            "prefix": c.prefix,
            "keywords": c.keywords,
            "token_count": c.token_count,
            "start_line": c.span.start_line,
            "end_line": c.span.end_line,
            "hash": _chunk_hash(profile_name, c),
            "embedding": c.embedding,
            "updated": now,
        }
        for c in chunks
    ]


def _write_parent(
    store: GraphStore, parent: _Parent, rows: list[dict[str, Any]], fingerprint: str
) -> None:
    store.delete_nodes("Chunk", where={"parent_id": parent.id})
    if rows:
        store.upsert_nodes("Chunk", rows)
        pk = "id" if parent.label == "Section" else "path"
        store.exec_write(
            f"MATCH (p:{parent.label} {{{pk}: $pid}}) "
            "UNWIND $ids AS cid MATCH (c:Chunk {id: cid}) "
            "MERGE (p)-[r:CONTAINS]->(c) SET r.origin = 'structural'",
            {"pid": parent.id, "ids": [r["id"] for r in rows]},
        )
        pairs = [
            {"a": a["id"], "b": b["id"]} for a, b in pairwise(rows) if a["profile"] == b["profile"]
        ]
        if pairs:
            store.exec_write(
                "UNWIND $pairs AS p MATCH (a:Chunk {id: p.a}), (b:Chunk {id: p.b}) "
                "MERGE (a)-[r:NEXT_SIBLING]->(b) SET r.origin = 'structural'",
                {"pairs": pairs},
            )
    pk = "id" if parent.label == "Section" else "path"
    store.exec_write(
        f"MATCH (p:{parent.label} {{{pk}: $pid}}) SET p.chunk_fingerprint = $fp",
        {"pid": parent.id, "fp": fingerprint},
    )


def _process_parent(
    parent: _Parent,
    corpus_cfg: CorpusConfig,
    deps: ChunkingDeps,
    store: GraphStore,
    cache: ParseCache,
    now: Any,
) -> tuple[int, int]:
    fingerprint = unit_fingerprint(deps.config_fp, parent.content_hash)
    if parent.stored_fp == fingerprint:
        return (0, 1)
    corpus = corpus_cfg.corpus.name
    suffix = Path(parent.path).suffix
    rows: list[dict[str, Any]] = []
    for profile in deps.config.profiles_for(suffix):
        built = _request_for(parent, corpus_cfg, cache, profile.name)
        if built is None:
            _log.debug("chunk: parent %s has no source on disk; skipping", parent.id)
            return (0, 1)
        req, breadcrumb, rel = built
        try:
            chunks = deps.chunker(suffix, profile.name).chunk(req)
        except Exception as exc:
            _log.warning(
                "chunk: strategy %s failed for %s (%s: %s); parent left unchunked",
                profile.strategy,
                parent.id,
                type(exc).__name__,
                exc,
            )
            return (0, 1)
        if not chunks:
            continue
        crumb = breadcrumb_text(breadcrumb, rel)
        if deps.config.prefix == "llm" and deps.inference is not None and deps.renderer is not None:
            prefixes = contextualise(
                deps.inference,
                deps.renderer,
                chunks,
                breadcrumb=crumb,
                document_summary=parent.summary or "",
            )
            for c, p in zip(chunks, prefixes, strict=True):
                c.prefix = p
        else:
            apply_prefix(
                chunks,
                static_prefix(
                    deps.config.prefix,
                    breadcrumb=breadcrumb,
                    rel_path=rel,
                    parent_summary=parent.summary,
                ),
            )
        shared = static_keywords(
            deps.config.augment_fulltext,
            key_points=parent.key_points,
            entities_mentioned=parent.entities_mentioned,
        )
        per_chunk: list[list[str]] | None = None
        if (
            "questions" in deps.config.augment_fulltext
            and deps.inference is not None
            and deps.renderer is not None
        ):
            per_chunk = generate_questions(deps.inference, deps.renderer, chunks, breadcrumb=crumb)
        apply_keywords(chunks, shared, per_chunk)
        pending = [c for c in chunks if c.embedding is None]
        if pending:
            if deps.embedder is None:
                raise RuntimeError("chunk phase requires an embedding provider")
            try:
                vectors = deps.embedder.embed([c.embed_text for c in pending])
            except Exception as exc:
                _log.warning(
                    "chunk: embedding failed for %s (%s: %s); parent left unchunked",
                    parent.id,
                    type(exc).__name__,
                    exc,
                )
                return (0, 1)
            for c, v in zip(pending, vectors, strict=True):
                c.embedding = v
        rows.extend(_rows_for(parent, profile.name, profile.strategy, chunks, corpus, now))
    _write_parent(store, parent, rows, fingerprint)
    return (1, 0)


def phase_chunk_units(
    corpus_cfg: CorpusConfig,
    deps: ChunkingDeps,
    store: GraphStore,
    *,
    concurrency: int = 1,
    parse_cache: ParseCache | None = None,
    paths: Sequence[Path] | None = None,
) -> PhaseResult:
    """(Re)chunk every parent whose fingerprint is stale; ``paths`` scopes to files."""
    corpus = corpus_cfg.corpus.name
    scope = {canonical_path(p) for p in paths} if paths is not None else None
    parents = _load_parents(store, corpus, scope)
    cache = parse_cache if parse_cache is not None else ParseCache(corpus_cfg)
    # Pre-populate the parse cache serially (workers only read it). A file
    # that vanished between enumeration and now is skipped here and again,
    # per parent, inside the worker.
    for p in {Path(x.path) for x in parents if x.label == "Section"}:
        try:
            cache.get(p)
        except OSError:
            _log.debug("chunk: source %s unreadable during cache prefill", p)
    now = dt.datetime.now(dt.UTC)

    def _worker(parent: _Parent) -> tuple[int, int]:
        return _process_parent(parent, corpus_cfg, deps, store, cache, now)

    processed, skipped = _parallel_map(parents, _worker, concurrency)
    return PhaseResult(name="chunk_units", processed=processed, skipped=skipped)


def chunk_units_for_path(
    path: Path,
    corpus_cfg: CorpusConfig,
    deps: ChunkingDeps,
    store: GraphStore,
    *,
    parse_cache: ParseCache | None = None,
) -> PhaseResult:
    return phase_chunk_units(corpus_cfg, deps, store, parse_cache=parse_cache, paths=[path])


def phase_gc_chunks(corpus_cfg: CorpusConfig, store: GraphStore) -> PhaseResult:
    """Delete orphaned chunks (parent gone) and chunks of profiles no longer configured."""
    corpus = corpus_cfg.corpus.name
    names = [p.name for p in corpus_cfg.chunking.profiles]
    orphans = store.exec_read(
        "MATCH (c:Chunk {corpus: $c}) WHERE NOT ()-[:CONTAINS]->(c) RETURN count(c) AS n",
        {"c": corpus},
    )
    stale_profiles = store.exec_read(
        "MATCH (c:Chunk {corpus: $c}) WHERE NOT c.profile IN $names RETURN count(c) AS n",
        {"c": corpus, "names": names},
    )
    n = int(orphans[0]["n"]) if orphans else 0
    m = int(stale_profiles[0]["n"]) if stale_profiles else 0
    if n:
        store.exec_write(
            "MATCH (c:Chunk {corpus: $c}) WHERE NOT ()-[:CONTAINS]->(c) DETACH DELETE c",
            {"c": corpus},
        )
    if m:
        store.exec_write(
            "MATCH (c:Chunk {corpus: $c}) WHERE NOT c.profile IN $names DETACH DELETE c",
            {"c": corpus, "names": names},
        )
    return PhaseResult(name="gc_chunks", processed=n + m, skipped=0)


def delete_chunks_for_path(store: GraphStore, path: str) -> None:
    """Remove a file's chunks before its File/Section nodes are deleted
    (DETACH DELETE on the parent would only orphan them)."""
    store.delete_nodes("Chunk", where={"path": path})


def estimate_chunks(
    corpus_cfg: CorpusConfig, deps: ChunkingDeps, files: Sequence[Path]
) -> dict[str, dict[str, int]]:
    """Dry run: per-profile chunk and embedding-token counts, no graph, no provider.

    Sections are parsed exactly as the pipeline would; strategies needing a
    provider (``semantic``, ``propositions``, ``late``) are approximated by
    ``structural`` so the estimate never makes a paid call.
    """
    from contextd.chunking.strategies.structural import StructuralStrategy
    from contextd.indexer.units import extractor_for

    structural = StructuralStrategy(deps.tokenizer)
    cache = ParseCache(corpus_cfg)
    out: dict[str, dict[str, int]] = {}
    llm_calls = 0
    for f in files:
        suffix = f.suffix
        requests: list[tuple[str, int, tuple[str, ...]]] = []
        if (
            extractor_for(corpus_cfg, suffix) is not None
            and corpus_cfg.corpus.granularity == "section"
        ):
            parsed = cache.get(f)
            for sec in parsed.sections:
                requests.append(
                    (sec.body, sec.start_line, (*parsed.parent_chain(sec.anchor), sec.title))
                )
        else:
            try:
                requests.append((f.read_text(encoding="utf-8", errors="replace"), 0, ()))
            except OSError:
                continue
        for profile in deps.config.profiles_for(suffix):
            stats = out.setdefault(profile.name, {"chunks": 0, "embed_tokens": 0, "llm_calls": 0})
            paid = profile.strategy in ("semantic", "propositions", "late")
            for text, base_line, crumb in requests:
                req = ChunkRequest(
                    text=text,
                    profile=profile,
                    blocks=corpus_cfg.chunking.blocks,
                    base_line=base_line,
                    breadcrumb=crumb,
                    suffix=suffix,
                )
                chunks = (
                    structural.chunk(req) if paid else deps.chunker(suffix, profile.name).chunk(req)
                )
                stats["chunks"] += len(chunks)
                stats["embed_tokens"] += sum(c.token_count for c in chunks) + len(chunks) * (
                    deps.tokenizer.count(breadcrumb_text(crumb, f.name))
                )
                if profile.strategy == "propositions":
                    stats["llm_calls"] += 1
        # Prefix / question generation is one call per parent unit regardless
        # of how many profiles exist (the phase batches all chunks of a parent).
        if deps.config.prefix == "llm":
            llm_calls += len(requests)
        if "questions" in deps.config.augment_fulltext:
            llm_calls += len(requests)
    if llm_calls:
        out.setdefault("_prefix_and_questions", {"chunks": 0, "embed_tokens": 0, "llm_calls": 0})[
            "llm_calls"
        ] = llm_calls
    return out


def mark_topics_dirty(store: GraphStore, corpus: str) -> None:
    """Flag the corpus for re-clustering; the daemon/bootstrap topic phase clears it."""
    store.exec_write("MATCH (n:Corpus {name: $c}) SET n.topics_dirty = true", {"c": corpus})


def config_drifted(store: GraphStore, corpus: str, config_fp: str) -> bool:
    """True when the stored chunking-config fingerprint differs from ``config_fp``.

    A corpus that has never been bootstrapped with chunking (no stored value)
    is not "drifted" - there is nothing stale to re-chunk; the next bootstrap
    stamps it.
    """
    rows = store.exec_read(
        "MATCH (n:Corpus {name: $c}) RETURN n.chunk_config_fingerprint AS fp", {"c": corpus}
    )
    stored = rows[0].get("fp") if rows else None
    return stored is not None and stored != config_fp


def stamp_config(store: GraphStore, corpus: str, config_fp: str) -> None:
    store.exec_write(
        "MATCH (n:Corpus {name: $c}) SET n.chunk_config_fingerprint = $fp",
        {"c": corpus, "fp": config_fp},
    )


def rechunk_corpus(
    corpus_cfg: CorpusConfig, deps: ChunkingDeps, store: GraphStore, *, concurrency: int = 1
) -> PhaseResult:
    """Whole-corpus re-chunk used by the daemon when the chunking config drifted.

    Fingerprint-gated per parent, so only parents whose config-dependent
    fingerprint changed do work; then the corpus-level fingerprint is stamped.
    """
    result = phase_chunk_units(corpus_cfg, deps, store, concurrency=concurrency)
    phase_gc_chunks(corpus_cfg, store)
    stamp_config(store, corpus_cfg.corpus.name, deps.config_fp)
    return result
