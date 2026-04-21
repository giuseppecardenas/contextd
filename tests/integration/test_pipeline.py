from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.integration


def test_bootstrap_creates_inferred_edges(backend, tmp_path: Path) -> None:
    """Verify phase_relate's delete_edges + upsert_edge with label kwargs works
    on both backends (spec-delta (c) coverage)."""
    from contextd.corpus_config import CorpusConfig
    from contextd.indexer.hasher import FileHasher
    from contextd.indexer.pipeline import run_bootstrap
    from contextd.inference.relate import InferredRelationship
    from contextd.inference.summarise import FileSummary

    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("alpha references beta")
    (corpus_root / "b.md").write_text("beta references alpha")

    corpus_cfg = CorpusConfig.model_validate(
        {
            "corpus": {"name": "test", "root": str(corpus_root), "include": ["*.md"]},
        }
    )

    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.1] * 1024, [0.2] * 1024]

    fake_summariser = MagicMock()
    fake_summariser.summarise.return_value = FileSummary(
        summary="stub", key_points=[], entities_mentioned=[]
    )

    # Each file emits one REFERENCES edge to the other file.
    a_path = str(corpus_root / "a.md")
    b_path = str(corpus_root / "b.md")

    def infer(content: str, known_entities: list[str]) -> list[InferredRelationship]:
        # Emit "a → b" when the content starts with "alpha", "b → a" otherwise.
        if content.startswith("alpha"):
            return [
                InferredRelationship(
                    edge_type="REFERENCES",
                    target_type="File",
                    target_name=b_path,
                    confidence=0.9,
                    reason="test",
                )
            ]
        return [
            InferredRelationship(
                edge_type="REFERENCES",
                target_type="File",
                target_name=a_path,
                confidence=0.9,
                reason="test",
            )
        ]

    fake_inferrer = MagicMock()
    fake_inferrer.infer.side_effect = infer

    run_bootstrap(
        corpus=corpus_cfg,
        store=backend,
        embedder=fake_embedder,
        summariser=fake_summariser,
        inferrer=fake_inferrer,
        hasher=FileHasher(),
        entity_sampler=lambda _s: [],
    )

    # Assert 2 inferred REFERENCES edges exist.
    rows = backend.exec_read("MATCH (:File)-[r:REFERENCES]->(:File) RETURN count(r) AS c")
    assert rows[0]["c"] == 2


def test_bootstrap_on_sample_corpus(backend, tmp_path: Path) -> None:
    """Run a full bootstrap against a fake corpus and both backends."""
    from contextd.corpus_config import CorpusConfig
    from contextd.indexer.hasher import FileHasher
    from contextd.indexer.pipeline import run_bootstrap

    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "a.md").write_text("alpha content mentioning beta")
    (corpus_root / "b.md").write_text("beta content mentioning alpha")

    corpus_cfg = CorpusConfig.model_validate(
        {
            "corpus": {"name": "test", "root": str(corpus_root), "include": ["*.md"]},
        }
    )

    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.1] * 1024, [0.2] * 1024]

    fake_summariser = MagicMock()
    from contextd.inference.summarise import FileSummary

    fake_summariser.summarise.return_value = FileSummary(
        summary="stub", key_points=[], entities_mentioned=[]
    )

    fake_inferrer = MagicMock()
    fake_inferrer.infer.return_value = []

    result = run_bootstrap(
        corpus=corpus_cfg,
        store=backend,
        embedder=fake_embedder,
        summariser=fake_summariser,
        inferrer=fake_inferrer,
        hasher=FileHasher(),
        entity_sampler=lambda _s: [],
    )

    phase_names = [p.name for p in result.phases]
    assert phase_names == ["enumerate", "embed", "summarise", "relate", "close"]
    assert result.phases[0].processed == 2  # two files enumerated
    assert result.phases[1].processed == 2  # two files embedded
    assert result.phases[2].processed == 2  # two files summarised

    corpus_nodes = backend.exec_read(
        "MATCH (n:Corpus {name: $c}) RETURN n.name AS name", {"c": "test"}
    )
    assert len(corpus_nodes) == 1
    assert corpus_nodes[0]["name"] == "test"


def test_section_granular_bootstrap(backend, tmp_path: Path) -> None:
    """Run a full section-granular bootstrap on both backends (spec §5.11.3)."""
    from contextd.corpus_config import CorpusConfig
    from contextd.indexer.hasher import FileHasher
    from contextd.indexer.pipeline import run_bootstrap
    from contextd.inference.summarise import FileSummary

    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "doc.md").write_text(
        "# Title\n\n## §1 First\n\nbody 1\n\n### §1.1 Sub\n\nbody 1.1\n\n## §2 Second\n\nbody 2\n"
    )

    cfg = CorpusConfig.model_validate(
        {
            "corpus": {
                "name": "sec",
                "root": str(corpus_root),
                "include": ["*.md"],
                "granularity": "section",
                "heading_min_level": 2,
                "heading_max_level": 4,
            },
        }
    )

    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.1] * 1024] * 3
    fake_summariser = MagicMock()
    fake_summariser.summarise.return_value = FileSummary(
        summary="s", key_points=[], entities_mentioned=[]
    )
    fake_inferrer = MagicMock()
    fake_inferrer.infer.return_value = []

    result = run_bootstrap(
        corpus=cfg,
        store=backend,
        embedder=fake_embedder,
        summariser=fake_summariser,
        inferrer=fake_inferrer,
        hasher=FileHasher(),
        entity_sampler=lambda _s: [],
    )

    phase_names = [p.name for p in result.phases]
    assert phase_names[0] == "enumerate_sections"
    assert "derive_file_level" in phase_names

    # Three sections emitted (§1 First, §1.1 Sub, §2 Second).
    sections = backend.exec_read("MATCH (s:Section) RETURN s.title AS title ORDER BY s.ordinal")
    titles = [r["title"] for r in sections]
    assert "§1 First" in titles
    assert "§1.1 Sub" in titles
    assert "§2 Second" in titles

    # Section summaries populated by phase_summarise_sections.
    summaries = backend.exec_read("MATCH (s:Section {corpus: 'sec'}) RETURN s.summary AS summary")
    assert len(summaries) == 3
    assert all(row["summary"] == "s" for row in summaries)

    # File summary populated by phase_derive_file_level (concatenated first
    # sentences of child section summaries, capped at 500 chars).
    file_rows = backend.exec_read("MATCH (f:File {corpus: 'sec'}) RETURN f.summary AS summary")
    assert len(file_rows) == 1
    assert file_rows[0]["summary"] is not None
    assert file_rows[0]["summary"] != ""

    # SD #73: File.hash is now a real MD5 (32 hex chars), not "__pending__".
    hash_rows = backend.exec_read("MATCH (f:File {corpus: 'sec'}) RETURN f.hash AS hash")
    assert len(hash_rows) == 1
    stored_hash = hash_rows[0]["hash"]
    assert stored_hash != "__pending__"
    assert len(stored_hash) == 32
    assert all(c in "0123456789abcdef" for c in stored_hash)


def test_section_granular_inferred_edges(backend, tmp_path: Path) -> None:
    """Exercise phase_relate_sections' delete_edges + upsert_edge with the
    Delta B label kwargs (src_label='Section', dst_label=...). Without this,
    the label-plumbing code path is not validated at runtime on Kuzu."""
    from contextd.corpus_config import CorpusConfig
    from contextd.indexer.hasher import FileHasher
    from contextd.indexer.pipeline import run_bootstrap
    from contextd.inference.relate import InferredRelationship
    from contextd.inference.summarise import FileSummary

    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "doc.md").write_text("# Title\n\n## §A\n\nbody a\n\n## §B\n\nbody b\n")

    cfg = CorpusConfig.model_validate(
        {
            "corpus": {
                "name": "rel_sec",
                "root": str(corpus_root),
                "include": ["*.md"],
                "granularity": "section",
                "heading_min_level": 2,
                "heading_max_level": 4,
            },
        }
    )

    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.1] * 1024] * 2
    fake_summariser = MagicMock()
    fake_summariser.summarise.return_value = FileSummary(
        summary="s", key_points=[], entities_mentioned=[]
    )

    # Each section emits one Section→Section REFERENCES edge to the other section.
    a_id = f"{corpus_root / 'doc.md'}#a"
    b_id = f"{corpus_root / 'doc.md'}#b"

    def infer(content: str, known_entities: list[str]) -> list[InferredRelationship]:
        if content.startswith("body a"):
            return [
                InferredRelationship(
                    edge_type="REFERENCES",
                    target_type="Section",
                    target_name=b_id,
                    confidence=0.9,
                    reason="test",
                )
            ]
        return [
            InferredRelationship(
                edge_type="REFERENCES",
                target_type="Section",
                target_name=a_id,
                confidence=0.9,
                reason="test",
            )
        ]

    fake_inferrer = MagicMock()
    fake_inferrer.infer.side_effect = infer

    run_bootstrap(
        corpus=cfg,
        store=backend,
        embedder=fake_embedder,
        summariser=fake_summariser,
        inferrer=fake_inferrer,
        hasher=FileHasher(),
        entity_sampler=lambda _s: [],
    )

    # Two Section→Section REFERENCES edges must exist.
    rows = backend.exec_read("MATCH (:Section)-[r:REFERENCES]->(:Section) RETURN count(r) AS c")
    assert rows[0]["c"] == 2
