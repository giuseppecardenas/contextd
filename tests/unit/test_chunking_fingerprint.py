from __future__ import annotations

from contextd.chunking import config_fingerprint, unit_fingerprint
from contextd.corpus_config import ChunkingSection, ChunkProfile


def test_fingerprint_is_stable_and_hex() -> None:
    a = config_fingerprint(ChunkingSection(), "words")
    b = config_fingerprint(ChunkingSection(), "words")
    assert a == b and len(a) == 64 and int(a, 16) >= 0


def test_fingerprint_changes_on_every_knob() -> None:
    base = config_fingerprint(ChunkingSection(), "words")
    variants = [
        config_fingerprint(ChunkingSection(), "voyage:voyage-4-large"),
        config_fingerprint(ChunkingSection(prefix="none"), "words"),
        config_fingerprint(ChunkingSection(augment_fulltext=[]), "words"),
        config_fingerprint(
            ChunkingSection(profiles=[ChunkProfile(name="fine", max_tokens=300)]), "words"
        ),
        config_fingerprint(
            ChunkingSection(profiles=[ChunkProfile(name="fine", overlap=0.1)]), "words"
        ),
    ]
    assert len({base, *variants}) == len(variants) + 1


def test_fingerprint_independent_of_profile_declaration_object_identity() -> None:
    s1 = ChunkingSection(profiles=[ChunkProfile(name="a"), ChunkProfile(name="b")])
    s2 = ChunkingSection(profiles=[ChunkProfile(name="a"), ChunkProfile(name="b")])
    assert config_fingerprint(s1, "words") == config_fingerprint(s2, "words")
    # Order is significant: profile order drives chunk ordinals/ids.
    s3 = ChunkingSection(profiles=[ChunkProfile(name="b"), ChunkProfile(name="a")])
    assert config_fingerprint(s1, "words") != config_fingerprint(s3, "words")


def test_unit_fingerprint_binds_config_and_content() -> None:
    cfg = config_fingerprint(ChunkingSection(), "words")
    assert unit_fingerprint(cfg, "h1") != unit_fingerprint(cfg, "h2")
    assert unit_fingerprint(cfg, "h1") != unit_fingerprint("other", "h1")
    assert unit_fingerprint(cfg, "h1") == unit_fingerprint(cfg, "h1")
