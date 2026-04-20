from pathlib import Path

import pytest

from contextd.corpus_config import CorpusConfig, CorpusConfigError


def test_loads_minimal_corpus(tmp_path: Path) -> None:
    (tmp_path / "corpus.toml").write_text("""
[corpus]
name = "notes"
root = "/home/alice/notes"
""")
    cfg = CorpusConfig.load(tmp_path / "corpus.toml")
    assert cfg.corpus.name == "notes"
    assert cfg.corpus.root == "/home/alice/notes"
    assert cfg.corpus.granularity == "file"  # default


def test_section_granularity(tmp_path: Path) -> None:
    (tmp_path / "corpus.toml").write_text("""
[corpus]
name = "prd"
root = "/home/alice/prd"
granularity = "section"
heading_min_level = 2
heading_max_level = 4
""")
    cfg = CorpusConfig.load(tmp_path / "corpus.toml")
    assert cfg.corpus.granularity == "section"
    assert cfg.corpus.heading_min_level == 2
    assert cfg.corpus.heading_max_level == 4


def test_rejects_reserved_auto(tmp_path: Path) -> None:
    (tmp_path / "corpus.toml").write_text("""
[corpus]
name = "x"
root = "/tmp/x"
granularity = "auto"
""")
    with pytest.raises(CorpusConfigError, match="'auto' is reserved"):
        CorpusConfig.load(tmp_path / "corpus.toml")


def test_ontology_aliases(tmp_path: Path) -> None:
    (tmp_path / "corpus.toml").write_text("""
[corpus]
name = "prd"
root = "/tmp/prd"

[ontology.aliases]
Registry = "Pattern"
FRRow = "Ticket"
""")
    cfg = CorpusConfig.load(tmp_path / "corpus.toml")
    assert cfg.ontology.aliases == {"Registry": "Pattern", "FRRow": "Ticket"}
