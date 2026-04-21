from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.integration


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
