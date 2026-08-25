"""``search`` / ``expand_chunk`` over a bootstrapped chunk graph (plan §8).

The corpus is seeded through the real pipeline (``_chunk_corpus.bootstrap``)
so the rows the tools return come from the same nodes, edges and indexes a
user's graph has. ``HashEmbedder`` gives the vector legs a real signal, so
``mode="hybrid"`` exercises fusion rather than degrading to full-text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from contextd._paths import canonical_path
from contextd.mcp import tools
from contextd.storage.base import GraphStore
from tests.integration._chunk_corpus import (
    BODY_ONLY_TOKEN,
    FAQ_TOKEN,
    SUMMARY,
    HashEmbedder,
    bootstrap,
    corpus_config,
    write_corpus,
)

pytestmark = pytest.mark.integration

_EVIDENCE_KEYS = {"chunk_id", "profile", "kind", "start_line", "end_line", "text"}


def _first_line(text: str) -> str:
    return next(line.strip() for line in text.splitlines() if line.strip())


def test_body_only_token_is_found_through_chunks_with_line_evidence(
    backend: GraphStore, tmp_path: Path
) -> None:
    """(a) the headline regression: a token that lives only in a section body
    is reachable; (b) chunk rows carry line-accurate evidence and neighbours."""
    root = tmp_path / "corpus"
    write_corpus(root)
    embedder = HashEmbedder()
    bootstrap(backend, corpus_config(root), embedder=embedder)
    guide = canonical_path(root / "guide.md")

    # Precondition: nothing the summary-based search legs index carries the
    # token, so the old ``kind="Section"`` search cannot find it ...
    assert (
        backend.exec_read(
            "MATCH (n) WHERE n.summary CONTAINS $t OR n.title CONTAINS $t RETURN count(n) AS n",
            {"t": BODY_ONLY_TOKEN},
        )[0]["n"]
        == 0
    )
    assert tools.search(backend, BODY_ONLY_TOKEN, kind="Section", mode="fulltext") == []

    # ... while the default (Chunk) search does, in both modes.
    for mode in ("fulltext", "hybrid"):
        rows = tools.search(
            backend,
            BODY_ONLY_TOKEN,
            embedder=embedder,
            mode=mode,  # type: ignore[arg-type]
            return_unit="chunk",
            window=1,
            limit=10,
            # The coarse Overview chunk is ~5 KB with the token past the
            # default 1200-char clip; widen it so the hit is visible.
            max_evidence_chars=20_000,
        )
        assert rows, mode
        top = rows[0]
        assert top["unit"] == "chunk" and top["path"] == guide, (mode, top)
        assert top["parent_id"] == f"{guide}#overview"
        assert BODY_ONLY_TOKEN in top["evidence"]["text"], (mode, top["evidence"])
    hits = [r for r in rows if BODY_ONLY_TOKEN in r["evidence"]["text"]]
    # Both profiles cover the Overview section, so both contribute a hit.
    assert {r["profile"] for r in hits} == {"fine", "coarse"}

    # Evidence lines are file coordinates: the slice they name contains the
    # evidence text. Checked on the ``fine`` chunk, whose text is verbatim
    # source (``coarse`` may prepend an overlap tail from the previous chunk).
    lines = (root / "guide.md").read_text(encoding="utf-8").splitlines()
    fine = next(r for r in hits if r["profile"] == "fine")
    ev = fine["evidence"]
    assert set(ev) >= _EVIDENCE_KEYS
    assert ev["chunk_id"] == fine["id"]
    assert 0 <= ev["start_line"] < ev["end_line"] <= len(lines)
    window_lines = [ln.strip() for ln in lines[ev["start_line"] : ev["end_line"]]]
    assert _first_line(ev["text"]) in window_lines, (ev, window_lines)
    assert any(BODY_ONLY_TOKEN in ln for ln in window_lines)
    # The token sits mid-paragraph, so the fine hit has neighbours both sides.
    assert ev["context_before"] and ev["context_after"]
    assert BODY_ONLY_TOKEN not in ev["context_before"] + ev["context_after"]
    # window=0 drops the neighbour context entirely.
    bare = tools.search(
        backend, BODY_ONLY_TOKEN, embedder=embedder, return_unit="chunk", window=0, limit=3
    )
    assert bare and "context_before" not in bare[0]["evidence"]
    assert all("embedding" not in r for r in rows)


def test_collapse_profiles_corpus_filter_and_expand(backend: GraphStore, tmp_path: Path) -> None:
    """(c) section collapse, (d) profile filters, (e) corpus filter, (f) expand_chunk."""
    root_a, root_b = tmp_path / "alpha", tmp_path / "beta"
    write_corpus(root_a)
    write_corpus(root_b)
    embedder = HashEmbedder()
    bootstrap(backend, corpus_config(root_a, "alpha"), embedder=embedder)
    bootstrap(backend, corpus_config(root_b, "beta"), embedder=embedder)
    guide_a = canonical_path(root_a / "guide.md")

    def search(query: str, *, mode: str = "hybrid", **kw: Any) -> list[dict[str, Any]]:
        return tools.search(
            backend,
            query,
            embedder=embedder,
            mode=mode,  # type: ignore[arg-type]
            limit=10,
            max_evidence_chars=20_000,
            **kw,
        )

    # (c) return_unit="section": one Section row per parent, best first,
    # with the number of distinct chunks that hit and the best as evidence.
    # Exact ``matched_chunks`` counts are asserted in fulltext mode: a vector
    # leg returns its k nearest chunks whatever their relevance, so under
    # hybrid fusion every chunk of a parent counts as matched.
    rows = search(BODY_ONLY_TOKEN, return_unit="section", corpus="alpha", mode="fulltext")
    assert rows and rows[0]["unit"] == "section"
    top = rows[0]
    assert top["id"] == f"{guide_a}#overview" and top["title"] == "Overview"
    assert top["corpus"] == "alpha" and top["summary"] == SUMMARY
    assert top["matched_chunks"] == 2  # one fine + one coarse chunk carry the token
    assert BODY_ONLY_TOKEN in top["evidence"]["text"]
    assert len({r["id"] for r in rows}) == len(rows), "a parent must collapse to one row"

    # return_unit="file" rolls the same hits up to the File.
    files = search(BODY_ONLY_TOKEN, return_unit="file", corpus="alpha", mode="fulltext")
    assert files[0]["unit"] == "file" and files[0]["path"] == guide_a
    assert files[0]["matched_chunks"] == 2
    # Hybrid fusion still ranks the Overview section first and keeps one
    # row per parent; only the member count is mode-dependent.
    hybrid = search(BODY_ONLY_TOKEN, return_unit="section", corpus="alpha")
    assert hybrid[0]["id"] == top["id"] and hybrid[0]["matched_chunks"] >= 2
    assert len({r["id"] for r in hybrid}) == len(hybrid)

    # (d) profile filters: every evidence chunk comes from the requested profile.
    for name in ("fine", "coarse"):
        rows = search(BODY_ONLY_TOKEN, return_unit="chunk", profiles=[name], corpus="alpha")
        assert rows, name
        assert {r["profile"] for r in rows} == {name}
        assert {r["evidence"]["profile"] for r in rows} == {name}
        assert BODY_ONLY_TOKEN in rows[0]["evidence"]["text"]
    fine_only = search(
        BODY_ONLY_TOKEN, return_unit="section", profiles=["fine"], corpus="alpha", mode="fulltext"
    )
    assert fine_only[0]["matched_chunks"] == 1

    # (e) corpus filter: both corpora hold identical text, so without a
    # filter both appear; with one, only the requested corpus's paths do.
    both = search(FAQ_TOKEN, return_unit="section")
    assert {r["corpus"] for r in both} == {"alpha", "beta"}
    for name, root in (("alpha", root_a), ("beta", root_b)):
        prefix = canonical_path(root)
        chunks = search(FAQ_TOKEN, return_unit="chunk", corpus=name)
        assert chunks and all(r["path"].startswith(prefix) for r in chunks), (name, chunks)
        sections = search(FAQ_TOKEN, return_unit="section", corpus=name)
        assert sections and {r["corpus"] for r in sections} == {name}
        assert sections[0]["title"] == "Installing"

    # (f) expand_chunk round-trips a chunk id with its neighbours and parent.
    hit = next(
        r
        for r in search(BODY_ONLY_TOKEN, return_unit="chunk", profiles=["fine"], corpus="alpha")
        if BODY_ONLY_TOKEN in r["evidence"]["text"]
    )
    expanded = tools.expand_chunk(backend, hit["id"], window=1)
    assert expanded is not None
    assert expanded["id"] == hit["id"] and expanded["profile"] == "fine"
    assert expanded["parent_id"] == f"{guide_a}#overview"
    assert expanded["parent_title"] == "Overview" and expanded["parent_summary"] == SUMMARY
    assert BODY_ONLY_TOKEN in expanded["text"]
    assert expanded["prefix"].startswith("Deployment Guide")
    assert (expanded["start_line"], expanded["end_line"]) == (
        hit["evidence"]["start_line"],
        hit["evidence"]["end_line"],
    )
    assert len(expanded["context_before"]) == 1 and len(expanded["context_after"]) == 1
    assert expanded["context_before"][0] == hit["evidence"]["context_before"]
    wide = tools.expand_chunk(backend, hit["id"], window=10)
    assert wide is not None
    assert len(wide["context_before"]) + len(wide["context_after"]) >= 3
    assert wide["context_before"][-1] == expanded["context_before"][0]
    assert tools.expand_chunk(backend, "no-such-chunk") is None
