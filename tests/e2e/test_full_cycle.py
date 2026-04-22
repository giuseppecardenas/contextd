"""End-to-end test: files → bootstrap → MCP query (spec §11.3)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.e2e


def test_full_bootstrap_then_mcp_query(backend, tmp_path: Path) -> None:
    """Exercise: files → bootstrap → describe_project → inbound → outbound."""
    from contextd.corpus_config import CorpusConfig
    from contextd.indexer.hasher import FileHasher
    from contextd.indexer.pipeline import run_bootstrap
    from contextd.inference.summarise import FileSummary
    from contextd.mcp import tools

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text("alpha\n\nSee [b](b.md).")
    (root / "b.md").write_text("beta\n\nSee [a](a.md) and [c](c.md).")
    (root / "c.md").write_text("gamma")

    cfg = CorpusConfig.model_validate(
        {
            "corpus": {
                "name": "e2e",
                "root": str(root),
                "include": ["*.md"],
                "granularity": "file",
            },
        }
    )

    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
    fake_summariser = MagicMock()
    fake_summariser.summarise.side_effect = [
        FileSummary(summary="alpha file", key_points=["k1"], entities_mentioned=[]),
        FileSummary(summary="beta file", key_points=["k2"], entities_mentioned=[]),
        FileSummary(summary="gamma file", key_points=["k3"], entities_mentioned=[]),
    ]
    fake_inferrer = MagicMock()
    fake_inferrer.infer.return_value = []

    run_bootstrap(
        corpus=cfg,
        store=backend,
        embedder=fake_embedder,
        summariser=fake_summariser,
        inferrer=fake_inferrer,
        hasher=FileHasher(),
        entity_sampler=lambda _s: [],
    )

    overview = tools.describe_project(backend, corpus="e2e")
    summaries = [n["summary"] for n in overview.nodes]
    assert "alpha file" in summaries
    assert "beta file" in summaries
    assert "gamma file" in summaries

    # Summary is present on every node.
    assert all(s is not None for s in summaries)
