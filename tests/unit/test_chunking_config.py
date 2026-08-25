"""[chunking] / [topics] corpus config models and the removed [embedding] keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from contextd.corpus_config import (
    ChunkingSection,
    ChunkProfile,
    ChunkProfileOverride,
    CorpusConfig,
    CorpusConfigError,
)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "corpus.toml"
    p.write_text(body, encoding="utf-8")
    return p


_HEADER = """
[corpus]
name = "notes"
root = "/home/alice/notes"
"""


def test_defaults_ship_fine_and_coarse_profiles(tmp_path: Path) -> None:
    cfg = CorpusConfig.load(_write(tmp_path, _HEADER))
    assert cfg.chunking.enabled is True
    assert [p.name for p in cfg.chunking.profiles] == ["fine", "coarse"]
    fine, coarse = cfg.chunking.profiles
    assert fine.max_tokens == 256 and fine.min_tokens == 48 and fine.overlap == 0.0
    assert coarse.max_tokens == 1024 and coarse.overlap_tokens == 102
    assert cfg.chunking.prefix == "breadcrumb"
    assert cfg.chunking.augment_fulltext == ["key_points"]
    assert cfg.topics.enabled is False


def test_removed_embedding_chunk_keys_rejected_with_migration_hint(tmp_path: Path) -> None:
    body = (
        _HEADER
        + """
[embedding]
chunk_tokens = 32000
chunk_overlap = 200
"""
    )
    with pytest.raises(CorpusConfigError, match=r"chunk_overlap, chunk_tokens were removed"):
        CorpusConfig.load(_write(tmp_path, body))


def test_profiles_parse_from_toml(tmp_path: Path) -> None:
    body = (
        _HEADER
        + """
[chunking]
tokenizer = "words"
prefix = "section_summary"

[[chunking.profiles]]
name = "sent"
strategy = "sentence_window"
window = 3

[[chunking.profiles]]
name = "big"
max_tokens = 2048
min_tokens = 500
overlap = 0.2
weight = 0.5

[chunking.blocks]
table_mode = "whole"

[chunking.suffix_overrides]
".txt" = { strategy = "recursive", max_tokens = 128 }
"""
    )
    cfg = CorpusConfig.load(_write(tmp_path, body))
    assert cfg.chunking.tokenizer == "words"
    assert cfg.chunking.profile("sent").strategy == "sentence_window"
    assert cfg.chunking.profile("big").weight == 0.5
    assert cfg.chunking.blocks.table_mode == "whole"
    overridden = cfg.chunking.profiles_for(".txt")
    assert all(p.strategy == "recursive" and p.max_tokens == 128 for p in overridden)
    assert overridden[1].min_tokens == 500 and overridden[1].overlap == 0.2  # untouched fields
    assert cfg.chunking.profiles_for(".md") == cfg.chunking.profiles


def test_duplicate_profile_names_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate profile names"):
        ChunkingSection(profiles=[ChunkProfile(name="a"), ChunkProfile(name="a")])


def test_enabled_without_profiles_rejected() -> None:
    with pytest.raises(ValueError, match="no profiles"):
        ChunkingSection(profiles=[])
    assert ChunkingSection(enabled=False, profiles=[]).profiles == []


@pytest.mark.parametrize("name", ["Fine", "1st", "with-dash", ""])
def test_profile_name_must_be_identifier_like(name: str) -> None:
    with pytest.raises(ValueError, match="must match"):
        ChunkProfile(name=name)


def test_min_tokens_must_be_below_max_for_size_bounded_strategies() -> None:
    with pytest.raises(ValueError, match="min_tokens"):
        ChunkProfile(name="x", max_tokens=100, min_tokens=100)
    # sentence_window ignores the size knobs, so the same values are accepted.
    ChunkProfile(name="x", strategy="sentence_window", max_tokens=100, min_tokens=100)


def test_semantic_threshold_ranges() -> None:
    with pytest.raises(ValueError, match="percentile threshold"):
        ChunkProfile(name="s", strategy="semantic", threshold=150.0)
    with pytest.raises(ValueError, match="stddev threshold"):
        ChunkProfile(name="s", strategy="semantic", threshold_type="stddev", threshold=0.0)
    ChunkProfile(name="s", strategy="semantic", threshold_type="iqr", threshold=1.5)


def test_suffix_override_key_must_be_suffix() -> None:
    with pytest.raises(ValueError, match="must be a file suffix"):
        ChunkingSection(suffix_overrides={"txt": ChunkProfileOverride(strategy="window")})


def test_override_apply_is_noop_when_empty() -> None:
    p = ChunkProfile(name="p")
    assert ChunkProfileOverride().apply(p) is p


def test_topics_section_parses(tmp_path: Path) -> None:
    body = (
        _HEADER
        + """
[topics]
enabled = true
max_layers = 2
soft_threshold = 0.2
"""
    )
    cfg = CorpusConfig.load(_write(tmp_path, body))
    assert cfg.topics.enabled and cfg.topics.max_layers == 2 and cfg.topics.soft_threshold == 0.2


def test_topics_rejects_out_of_range(tmp_path: Path) -> None:
    body = _HEADER + "\n[topics]\nmax_layers = 9\n"
    with pytest.raises(CorpusConfigError):
        CorpusConfig.load(_write(tmp_path, body))
