"""Retrieval-chunk phases end-to-end against Neo4j (plan §8, integration).

Each test bootstraps the synthetic corpus from ``_chunk_corpus`` through
``run_bootstrap(chunking=...)`` and inspects the graph directly. Several
scenarios share one container where they build on each other, since every
test pays for its own Neo4j start-up.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from contextd._paths import canonical_path
from contextd.indexer import phases_chunks
from contextd.indexer.hasher import FileHasher
from contextd.indexer.pipeline import run_incremental_file
from contextd.inference.relate import InferredRelationship
from contextd.storage.base import GraphStore
from tests.integration._chunk_corpus import (
    FAQ_SECTIONS,
    GUIDE_SECTIONS,
    SUMMARY,
    HashEmbedder,
    bootstrap,
    chunk_rows,
    chunking_deps,
    corpus_config,
    fake_summariser,
    phase,
    relate_deps,
    section_fingerprints,
    write_corpus,
)

pytestmark = pytest.mark.integration

_N_SECTIONS = len(GUIDE_SECTIONS) + len(FAQ_SECTIONS)


def _by_parent_profile(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["parent_id"], r["profile"])].append(r)
    return groups


def test_bootstrap_chunks_section_corpus_and_gates_on_fingerprint(
    backend: GraphStore, tmp_path: Path
) -> None:
    """(a) bootstrap shape + (b) a second bootstrap is a no-op for chunks."""
    root = tmp_path / "corpus"
    write_corpus(root)
    cfg = corpus_config(root)
    # The chunk phase gets its own embedder so its call count is isolated from
    # section enumeration, which re-embeds every section on each bootstrap.
    chunk_embedder = HashEmbedder()
    deps = chunking_deps(cfg, chunk_embedder)

    result, _ = bootstrap(backend, cfg, chunking=deps)

    # Phase accounting: every section is a chunk parent and none was skipped.
    units = phase(result, "chunk_units")
    assert (units.processed, units.skipped) == (_N_SECTIONS, 0)
    assert phase(result, "gc_chunks").processed == 0

    rows = chunk_rows(backend, "chunks")
    assert rows, "bootstrap wrote no Chunk nodes"
    assert {r["profile"] for r in rows} == {"fine", "coarse"}
    assert all(r["parent_label"] == "Section" for r in rows)

    # Size invariant: token_count <= max_tokens, plus the profile's overlap
    # budget (the overlap tail is prepended *after* packing, by design — see
    # ``test_structural_overlap_prepends_previous_tail``).
    for r in rows:
        profile = cfg.chunking.profile(r["profile"])
        assert r["token_count"] <= profile.max_tokens + profile.overlap_tokens, r["id"]
        assert r["id"] == phases_chunks.chunk_id(r["parent_id"], r["profile"], r["ordinal"])
        assert r["path"] and r["start_line"] >= 0 and r["end_line"] >= r["start_line"]

    # The block shapes the structural strategy handles specially all show up:
    # a fence stays a "code" chunk, the oversize table is sliced into parts
    # under "fine" (header repeated), the long paragraph splits into several.
    kinds = {r["kind"] for r in rows}
    assert {"code", "table"} <= kinds, kinds
    # Slice 1 packs with the section heading into a ``mixed`` chunk; the
    # later slice(s) stand alone as ``table`` chunks that repeat the header.
    table_slices = [r for r in rows if r["kind"] == "table" and r["profile"] == "fine"]
    assert table_slices
    assert all(
        r["part"] >= 2 and r["text"].startswith("| name | type | default |") for r in table_slices
    )
    overview_id = f"{canonical_path(root / 'guide.md')}#overview"
    groups = _by_parent_profile(rows)
    assert len(groups[(overview_id, "fine")]) > 1
    assert len(groups[(overview_id, "fine")]) > len(groups[(overview_id, "coarse")])

    # Every Section parent is stamped with a fingerprint.
    fps = section_fingerprints(backend, "chunks")
    assert len(fps) == _N_SECTIONS
    assert all(fp for fp in fps.values()), fps

    # CONTAINS: one structural edge per chunk, from the node it names as parent.
    contains = backend.exec_read(
        "MATCH (p:Section)-[r:CONTAINS]->(c:Chunk {corpus: $c}) "
        "RETURN p.id AS pid, c.parent_id AS parent_id, r.origin AS origin",
        {"c": "chunks"},
    )
    assert len(contains) == len(rows)
    assert all(e["pid"] == e["parent_id"] and e["origin"] == "structural" for e in contains)

    # NEXT_SIBLING: exactly the consecutive-ordinal pairs within each
    # (parent, profile), nothing across parents or profiles.
    expected_pairs: set[tuple[str, str]] = set()
    for members in groups.values():
        assert [m["ordinal"] for m in members] == list(range(len(members)))
        expected_pairs.update((a["id"], b["id"]) for a, b in pairwise(members))
    siblings = backend.exec_read(
        "MATCH (a:Chunk {corpus: $c})-[r:NEXT_SIBLING]->(b:Chunk) "
        "RETURN a.id AS a, b.id AS b, r.origin AS origin",
        {"c": "chunks"},
    )
    assert {(s["a"], s["b"]) for s in siblings} == expected_pairs
    assert expected_pairs, "no multi-chunk parent, the sibling chain is untested"
    assert all(s["origin"] == "structural" for s in siblings)

    # Corpus accounting written by phase_close.
    corpus_node = backend.exec_read(
        "MATCH (n:Corpus {name: $c}) RETURN n.chunk_count AS n, n.chunk_config_fingerprint AS fp",
        {"c": "chunks"},
    )[0]
    assert corpus_node["n"] == len(rows)
    assert corpus_node["fp"] == deps.config_fp

    # (b) Second bootstrap: the fingerprint gate skips every parent, nothing
    # is re-embedded, and the chunk set is byte-for-byte the same.
    embed_calls = chunk_embedder.calls
    result2, _ = bootstrap(backend, cfg, chunking=deps)
    units2 = phase(result2, "chunk_units")
    assert (units2.processed, units2.skipped) == (0, _N_SECTIONS)
    rows2 = chunk_rows(backend, "chunks")
    assert [(r["id"], r["hash"]) for r in rows2] == [(r["id"], r["hash"]) for r in rows]
    assert section_fingerprints(backend, "chunks") == fps
    assert chunk_embedder.calls == embed_calls


def test_incremental_rechunks_only_the_edited_section_and_deletes_with_file(
    backend: GraphStore, tmp_path: Path
) -> None:
    """(c) ``run_incremental_file`` re-chunks the touched section only; a
    deleted file takes its chunks with it."""
    root = tmp_path / "corpus"
    write_corpus(root)
    cfg = corpus_config(root)
    embedder = HashEmbedder()
    _, deps = bootstrap(backend, cfg, embedder=embedder)

    guide = root / "guide.md"
    notes_id = f"{canonical_path(guide)}#notes"
    before = {r["id"]: r["hash"] for r in chunk_rows(backend, "chunks")}
    fps_before = section_fingerprints(backend, "chunks")
    assert any(cid.startswith(notes_id + "~") for cid in before)

    # Edit the *last* section so no other section's line numbers move.
    guide.write_text(
        guide.read_text(encoding="utf-8").replace(
            "Keep this section short.", "Keep this section short, but mention the xyzzy knob."
        ),
        encoding="utf-8",
    )
    outcome = run_incremental_file(
        guide,
        cfg,
        backend,
        FileHasher(),
        embedder,
        fake_summariser(),
        relate_deps(),
        chunking=deps,
    )
    assert outcome.action == "indexed"

    after = {r["id"]: r["hash"] for r in chunk_rows(backend, "chunks")}
    fps_after = section_fingerprints(backend, "chunks")
    untouched = {cid for cid in before if not cid.startswith(notes_id + "~")}
    assert {cid: after[cid] for cid in untouched} == {cid: before[cid] for cid in untouched}
    assert {k: v for k, v in fps_after.items() if k != notes_id} == {
        k: v for k, v in fps_before.items() if k != notes_id
    }
    assert fps_after[notes_id] != fps_before[notes_id]
    notes_after = {cid: h for cid, h in after.items() if cid.startswith(notes_id + "~")}
    assert notes_after and all(before.get(cid) != h for cid, h in notes_after.items())
    hits = backend.exec_read(
        "MATCH (c:Chunk {parent_id: $p}) WHERE c.text CONTAINS 'xyzzy' RETURN count(c) AS n",
        {"p": notes_id},
    )
    assert hits[0]["n"] == len(notes_after)

    # Deleting a file removes its chunks and leaves the other file's alone.
    faq = root / "faq.md"
    faq_path = canonical_path(faq)
    assert (
        backend.exec_read("MATCH (c:Chunk {path: $p}) RETURN count(c) AS n", {"p": faq_path})[0][
            "n"
        ]
        > 0
    )
    faq.unlink()
    outcome = run_incremental_file(
        faq, cfg, backend, FileHasher(), embedder, fake_summariser(), relate_deps(), chunking=deps
    )
    assert outcome.action == "deleted"
    assert (
        backend.exec_read("MATCH (c:Chunk {path: $p}) RETURN count(c) AS n", {"p": faq_path})[0][
            "n"
        ]
        == 0
    )
    assert (
        backend.exec_read("MATCH (n) WHERE n.path = $p RETURN count(n) AS n", {"p": faq_path})[0][
            "n"
        ]
        == 0
    )
    remaining = chunk_rows(backend, "chunks")
    assert remaining and all(r["path"] == canonical_path(guide) for r in remaining)
    assert len(remaining) == len(after) - sum(1 for cid in after if faq_path in cid)


def test_profile_change_rechunks_everything_but_leaves_llm_layers_alone(
    backend: GraphStore, tmp_path: Path
) -> None:
    """(d) a ``max_tokens`` change re-fingerprints every parent; summaries and
    inferred edges are neither recomputed nor dropped."""
    root = tmp_path / "corpus"
    write_corpus(root)
    faq_path = canonical_path(root / "faq.md")
    installing_id, upgrading_id = f"{faq_path}#installing", f"{faq_path}#upgrading"

    def infer(
        content: str, *, identity: Any = None, candidates: Any = None
    ) -> list[InferredRelationship]:
        if content.startswith("## Installing"):
            return [
                InferredRelationship(
                    edge_type="REFERENCES",
                    target_type="Section",
                    target_name=upgrading_id,
                    confidence=0.9,
                    reason="test",
                )
            ]
        return []

    inferrer = MagicMock()
    inferrer.infer.side_effect = infer
    cfg = corpus_config(root)
    embedder = HashEmbedder()
    _, deps = bootstrap(
        backend,
        cfg,
        embedder=embedder,
        summariser=fake_summariser("first summary"),
        relate=relate_deps(inferrer),
    )

    def edges() -> list[dict[str, Any]]:
        return backend.exec_read(
            "MATCH (a:Section {id: $a})-[r:REFERENCES {origin: 'inferred'}]->(b:Section {id: $b}) "
            "RETURN r.confidence AS confidence",
            {"a": installing_id, "b": upgrading_id},
        )

    def summaries() -> dict[str, str]:
        return {
            r["id"]: r["summary"]
            for r in backend.exec_read(
                "MATCH (s:Section {corpus: 'chunks'}) RETURN s.id AS id, s.summary AS summary"
            )
        }

    assert len(edges()) == 1
    summaries_before = summaries()
    assert set(summaries_before.values()) <= {"first summary", "rolled first summary"}
    fps_before = section_fingerprints(backend, "chunks")
    rows_before = chunk_rows(backend, "chunks")
    fine_before = sum(1 for r in rows_before if r["profile"] == "fine")

    # Halve ``fine``; keep ``coarse`` at its default. New deps carry a new
    # config fingerprint, and the LLM collaborators are fresh mocks that
    # would produce *different* output if they were consulted.
    cfg2 = corpus_config(
        root,
        profiles=[
            {"name": "fine", "max_tokens": 128, "min_tokens": 24},
            {"name": "coarse", "max_tokens": 1024, "min_tokens": 200, "overlap": 0.1},
        ],
    )
    deps2 = chunking_deps(cfg2, embedder)
    assert deps2.config_fp != deps.config_fp
    summariser2 = fake_summariser("second summary")
    inferrer2 = MagicMock()
    inferrer2.infer.return_value = []
    result2, _ = bootstrap(
        backend,
        cfg2,
        embedder=embedder,
        summariser=summariser2,
        relate=relate_deps(inferrer2),
        chunking=deps2,
    )

    units = phase(result2, "chunk_units")
    assert (units.processed, units.skipped) == (_N_SECTIONS, 0)
    fps_after = section_fingerprints(backend, "chunks")
    assert set(fps_after) == set(fps_before)
    assert all(fps_after[k] != fps_before[k] for k in fps_before)
    rows_after = chunk_rows(backend, "chunks")
    assert sum(1 for r in rows_after if r["profile"] == "fine") > fine_before
    for r in rows_after:
        profile = cfg2.chunking.profile(r["profile"])
        assert r["token_count"] <= profile.max_tokens + profile.overlap_tokens, r["id"]
    corpus_fp = backend.exec_read(
        "MATCH (n:Corpus {name: 'chunks'}) RETURN n.chunk_config_fingerprint AS fp"
    )[0]["fp"]
    assert corpus_fp == deps2.config_fp

    # LLM layers untouched: same summaries, same inferred edge, no new calls.
    assert summaries() == summaries_before
    assert len(edges()) == 1
    assert summariser2.summarise.call_count == 0
    assert inferrer2.infer.call_count == 0


def test_refresh_chunks_refills_and_gc_drops_unconfigured_profile(
    backend: GraphStore, tmp_path: Path
) -> None:
    """(e) ``refresh="chunks"`` wipes + refills; ``phase_gc_chunks`` reaps a
    profile that is no longer configured."""
    root = tmp_path / "corpus"
    write_corpus(root)
    cfg = corpus_config(root)
    chunk_embedder = HashEmbedder()
    deps = chunking_deps(cfg, chunk_embedder)
    bootstrap(backend, cfg, chunking=deps)
    rows_before = chunk_rows(backend, "chunks")
    embed_calls = chunk_embedder.calls

    result, _ = bootstrap(backend, cfg, chunking=deps, refresh="chunks")
    units = phase(result, "chunk_units")
    # Every parent lost its fingerprint in the wipe, so every one is redone
    # (and re-embedded); the chunk set itself is deterministic.
    assert (units.processed, units.skipped) == (_N_SECTIONS, 0)
    assert chunk_embedder.calls > embed_calls
    rows_after = chunk_rows(backend, "chunks")
    assert [(r["id"], r["hash"]) for r in rows_after] == [(r["id"], r["hash"]) for r in rows_before]
    assert backend.exec_read(
        "MATCH (:Section)-[:CONTAINS]->(c:Chunk {corpus: 'chunks'}) RETURN count(c) AS n"
    )[0]["n"] == len(rows_after)

    # GC against a config that dropped ``coarse``: only its chunks go.
    coarse = sum(1 for r in rows_after if r["profile"] == "coarse")
    assert coarse > 0
    cfg_fine_only = corpus_config(
        root, profiles=[{"name": "fine", "max_tokens": 256, "min_tokens": 48}]
    )
    gc = phases_chunks.phase_gc_chunks(cfg_fine_only, backend)
    assert gc.processed == coarse
    remaining = chunk_rows(backend, "chunks")
    assert len(remaining) == len(rows_after) - coarse
    assert {r["profile"] for r in remaining} == {"fine"}
    # A second GC pass finds nothing left to reap.
    assert phases_chunks.phase_gc_chunks(cfg_fine_only, backend).processed == 0


def test_file_granular_txt_corpus_chunks_under_file_nodes(
    backend: GraphStore, tmp_path: Path
) -> None:
    """(f) non-markdown, file-granular corpora get ``File``-parented chunks."""
    root = tmp_path / "corpus"
    root.mkdir()
    body = " ".join(f"Line {i} of the plain text notes talks about item {i}." for i in range(60))
    (root / "one.txt").write_text(body + "\n", encoding="utf-8")
    (root / "two.txt").write_text("A short second note.\n", encoding="utf-8")
    cfg = corpus_config(root, "plain", granularity="file", include=("*.txt",))

    result, deps = bootstrap(backend, cfg)
    units = phase(result, "chunk_units")
    assert (units.processed, units.skipped) == (2, 0)

    rows = chunk_rows(backend, "plain")
    assert rows
    paths = {canonical_path(root / n) for n in ("one.txt", "two.txt")}
    assert {r["path"] for r in rows} == paths
    assert all(r["parent_label"] == "File" and r["parent_id"] == r["path"] for r in rows)
    assert {r["profile"] for r in rows} == {"fine", "coarse"}
    # The long file splits under ``fine``; the short one is a single chunk.
    groups = _by_parent_profile(rows)
    assert len(groups[(canonical_path(root / "one.txt"), "fine")]) > 1
    assert len(groups[(canonical_path(root / "two.txt"), "fine")]) == 1
    assert all(r["start_line"] == 0 for r in groups[(canonical_path(root / "two.txt"), "fine")])

    contains = backend.exec_read(
        "MATCH (f:File)-[r:CONTAINS]->(c:Chunk {corpus: 'plain'}) "
        "RETURN f.path AS fp, c.parent_id AS pid, r.origin AS origin"
    )
    assert len(contains) == len(rows)
    assert all(e["fp"] == e["pid"] and e["origin"] == "structural" for e in contains)

    files = backend.exec_read(
        "MATCH (f:File {corpus: 'plain'}) RETURN f.path AS path, f.chunk_fingerprint AS fp, "
        "f.summary AS summary"
    )
    assert {f["path"] for f in files} == paths
    assert all(f["fp"] for f in files)
    assert all(f["summary"] == SUMMARY for f in files)
    corpus_node = backend.exec_read(
        "MATCH (n:Corpus {name: 'plain'}) RETURN n.chunk_count AS n, n.chunk_config_fingerprint AS fp"
    )[0]
    assert corpus_node["n"] == len(rows)
    assert corpus_node["fp"] == deps.config_fp
