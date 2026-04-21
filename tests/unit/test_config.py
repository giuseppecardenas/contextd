from pathlib import Path

import pytest

from contextd.config import Config, ConfigError


def test_load_default_returns_valid_config() -> None:
    cfg = Config.load_default()
    assert cfg.providers.inference == "gemini"
    assert cfg.providers.embedding == "voyage"
    assert cfg.storage.backend == "neo4j"
    assert cfg.storage.memgraph.port == 7687
    assert cfg.storage.kuzu.db_path.endswith("/graph/")
    assert cfg.inference.summary_max_words == 100
    assert cfg.indexer.debounce_seconds == 30


def test_inference_summary_max_words_override(tmp_path: Path) -> None:
    user_cfg = tmp_path / "config.toml"
    user_cfg.write_text("""
[inference]
summary_max_words = 200
""")
    cfg = Config.load(user_cfg)
    assert cfg.inference.summary_max_words == 200


def test_load_user_overrides_defaults(tmp_path: Path) -> None:
    user_cfg = tmp_path / "config.toml"
    user_cfg.write_text("""
[storage]
backend = "kuzu"

[indexer]
debounce_seconds = 15
""")
    cfg = Config.load(user_cfg)
    assert cfg.storage.backend == "kuzu"
    assert cfg.indexer.debounce_seconds == 15
    # Unspecified fields fall back to defaults.
    assert cfg.providers.inference == "gemini"


def test_rejects_unknown_backend(tmp_path: Path) -> None:
    user_cfg = tmp_path / "config.toml"
    user_cfg.write_text("""
[storage]
backend = "redis"
""")
    with pytest.raises(ConfigError, match=r"memgraph.*kuzu"):
        Config.load(user_cfg)


def test_rejects_unknown_safety_block(tmp_path: Path) -> None:
    user_cfg = tmp_path / "config.toml"
    user_cfg.write_text("""
[providers.gemini]
safety_block = "BLOCK_EVERYTHING"
""")
    with pytest.raises(ConfigError, match="safety_block"):
        Config.load(user_cfg)
