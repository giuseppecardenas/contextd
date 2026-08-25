"""End-to-end test: files → bootstrap → MCP query (spec §11.3)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.e2e


def test_full_bootstrap_then_mcp_query(backend, tmp_path: Path) -> None:
    """Exercise: files → bootstrap → describe_project → search evidence → expand_chunk."""
    from contextd.config import Config
    from contextd.corpus_config import CorpusConfig
    from contextd.indexer.chunk_deps import build_chunking_deps
    from contextd.indexer.hasher import FileHasher
    from contextd.indexer.phases import RelateDeps
    from contextd.indexer.pipeline import run_bootstrap
    from contextd.inference.context import EmptyRetriever
    from contextd.inference.summarise import FileSummary
    from contextd.mcp import tools
    from contextd.mcp_server import _dispatch_tool

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text("alpha\n\nSee [b](b.md).")
    (root / "b.md").write_text("beta\n\nSee [a](a.md) and [c](c.md).")
    (root / "c.md").write_text("gamma", encoding="utf-8")

    cfg = CorpusConfig.model_validate(
        {
            "corpus": {
                "name": "e2e",
                "root": str(root),
                "include": ["*.md"],
                "granularity": "file",
            },
            # The word tokenizer keeps the chunk phase offline and deterministic.
            "chunking": {"tokenizer": "words"},
        }
    )

    # One vector per input: the chunk phase embeds in batches of its own size.
    fake_embedder = MagicMock()
    fake_embedder.embed.side_effect = lambda texts: [[0.1] * 1024 for _ in texts]
    fake_summariser = MagicMock()
    fake_summariser.roll_up.return_value = "rolled"

    # Content-keyed, not call-ordered: phase_summarise visits files in
    # filesystem enumeration order, which differs between Windows and the
    # Linux CI runner, so a positional side_effect assigned "beta file" to
    # c.md on CI and the parent_summary assertion below failed there.
    def _summarise(content: str, *, context: object = None) -> FileSummary:
        word = content.split()[0]
        return FileSummary(summary=f"{word} file", key_points=[f"k-{word}"], entities_mentioned=[])

    fake_summariser.summarise.side_effect = _summarise
    fake_inferrer = MagicMock()
    fake_inferrer.infer.return_value = []

    run_bootstrap(
        corpus=cfg,
        store=backend,
        embedder=fake_embedder,
        summariser=fake_summariser,
        relate=RelateDeps(inferrer=fake_inferrer, retriever=EmptyRetriever()),
        hasher=FileHasher(),
        chunking=build_chunking_deps(
            Config(), cfg, embedder=fake_embedder, inference=None, renderer=None
        ),
    )

    overview = tools.describe_project(backend, corpus="e2e")
    summaries = [n["summary"] for n in overview.nodes]
    assert "alpha file" in summaries
    assert "beta file" in summaries
    assert "gamma file" in summaries

    # Summary is present on every node.
    assert all(s is not None for s in summaries)

    # Chunk search through the MCP dispatcher returns evidence rows, and the
    # evidence chunk id round-trips through expand_chunk.
    hits = json.loads(
        _dispatch_tool(
            "search", {"query": "gamma", "return_unit": "chunk"}, backend, embedder=fake_embedder
        )[0]["text"]
    )
    assert hits and all("evidence" in h for h in hits)
    assert hits[0]["evidence"]["text"].startswith("gamma")
    expanded = json.loads(
        _dispatch_tool("expand_chunk", {"chunk_id": hits[0]["evidence"]["chunk_id"]}, backend)[0][
            "text"
        ]
    )
    assert expanded["id"] == hits[0]["evidence"]["chunk_id"]
    assert expanded["parent_summary"] == "gamma file"
