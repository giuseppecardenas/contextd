from pathlib import Path

import pytest
from pydantic import ValidationError

from contextd.config import Config, ConfigError, IndexerConfig


def test_indexer_config_allowed_branches_defaults_empty() -> None:
    from contextd.config import IndexerConfig

    assert IndexerConfig().allowed_branches == []


def test_indexer_config_allowed_branches_parsed() -> None:
    from contextd.config import IndexerConfig

    cfg = IndexerConfig(allowed_branches=["main", "develop"])
    assert cfg.allowed_branches == ["main", "develop"]


def test_indexer_config_incremental_workers_defaults_to_4() -> None:
    from contextd.config import IndexerConfig

    assert IndexerConfig().incremental_workers == 4


def test_indexer_config_incremental_workers_rejects_zero() -> None:
    from contextd.config import IndexerConfig

    with pytest.raises(ValidationError):
        IndexerConfig(incremental_workers=0)


def test_logging_config_rotation_defaults() -> None:
    from contextd.config import LoggingConfig

    cfg = LoggingConfig()
    assert cfg.max_log_bytes == 10_485_760
    assert cfg.log_backup_count == 5


def test_logging_config_max_log_bytes_zero_is_valid() -> None:
    from contextd.config import LoggingConfig

    cfg = LoggingConfig(max_log_bytes=0)
    assert cfg.max_log_bytes == 0


def test_load_default_returns_valid_config() -> None:
    cfg = Config.load_default()
    assert cfg.providers.summary == "gemini"
    assert cfg.providers.inference == "gemini"
    assert cfg.providers.translation == "gemini"
    assert cfg.providers.embedding == "voyage"
    assert cfg.storage.backend == "neo4j"
    assert cfg.storage.memgraph.port == 7687
    assert cfg.storage.neo4j.port == 7687
    assert cfg.inference.summary_max_words == 100
    assert cfg.indexer.debounce_seconds == 30
    assert cfg.indexer.inference_concurrency == 1


def test_providers_config_per_call_site_defaults_to_gemini() -> None:
    from contextd.config import ProvidersConfig

    pcfg = ProvidersConfig()
    assert pcfg.summary == "gemini"
    assert pcfg.inference == "gemini"
    assert pcfg.translation == "gemini"


def test_providers_config_summary_can_be_openai_compat(tmp_path: Path) -> None:
    user_cfg = tmp_path / "config.toml"
    user_cfg.write_text("""
[providers]
summary = "openai_compat"
inference = "gemini"
translation = "gemini"
""")
    cfg = Config.load(user_cfg)
    assert cfg.providers.summary == "openai_compat"
    assert cfg.providers.inference == "gemini"
    assert cfg.providers.translation == "gemini"


def test_providers_config_rejects_unknown_provider_name(tmp_path: Path) -> None:
    user_cfg = tmp_path / "config.toml"
    user_cfg.write_text("""
[providers]
summary = "anthropic"
""")
    with pytest.raises(ConfigError, match=r"gemini.*openai_compat|openai_compat.*gemini"):
        Config.load(user_cfg)


def test_openai_compat_config_defaults_loadable() -> None:
    from contextd.config import OpenAICompatConfig

    cfg = OpenAICompatConfig()
    assert cfg.base_url == "http://localhost:11434/v1"
    assert cfg.api_key_env is None
    assert cfg.json_mode is True
    assert cfg.request_timeout_seconds == 120.0


def test_openai_compat_config_rejects_zero_timeout() -> None:
    from contextd.config import OpenAICompatConfig

    with pytest.raises(ValidationError):
        OpenAICompatConfig(request_timeout_seconds=0.0)


def test_inference_concurrency_override(tmp_path: Path) -> None:
    user_cfg = tmp_path / "config.toml"
    user_cfg.write_text("""
[indexer]
inference_concurrency = 7
""")
    cfg = Config.load(user_cfg)
    assert cfg.indexer.inference_concurrency == 7


def test_inference_concurrency_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        IndexerConfig(inference_concurrency=0)


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
backend = "memgraph"

[indexer]
debounce_seconds = 15
""")
    cfg = Config.load(user_cfg)
    assert cfg.storage.backend == "memgraph"
    assert cfg.indexer.debounce_seconds == 15
    # Unspecified fields fall back to defaults.
    assert cfg.providers.summary == "gemini"
    assert cfg.providers.inference == "gemini"
    assert cfg.providers.translation == "gemini"


def test_rejects_unknown_backend(tmp_path: Path) -> None:
    user_cfg = tmp_path / "config.toml"
    user_cfg.write_text("""
[storage]
backend = "redis"
""")
    with pytest.raises(ConfigError, match=r"memgraph.*neo4j"):
        Config.load(user_cfg)


def test_rejects_unknown_safety_block(tmp_path: Path) -> None:
    user_cfg = tmp_path / "config.toml"
    user_cfg.write_text("""
[providers.gemini]
safety_block = "BLOCK_EVERYTHING"
""")
    with pytest.raises(ConfigError, match="safety_block"):
        Config.load(user_cfg)
