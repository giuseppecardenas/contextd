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
[storage.neo4j]
port = 7999

[indexer]
debounce_seconds = 15
""")
    cfg = Config.load(user_cfg)
    assert cfg.storage.neo4j.port == 7999
    assert cfg.indexer.debounce_seconds == 15
    # Unspecified fields fall back to defaults.
    assert cfg.storage.backend == "neo4j"
    assert cfg.providers.summary == "gemini"
    assert cfg.providers.inference == "gemini"
    assert cfg.providers.translation == "gemini"


def test_rejects_unknown_backend(tmp_path: Path) -> None:
    user_cfg = tmp_path / "config.toml"
    user_cfg.write_text("""
[storage]
backend = "redis"
""")
    with pytest.raises(ConfigError, match=r"neo4j"):
        Config.load(user_cfg)


def test_rejects_unknown_safety_block(tmp_path: Path) -> None:
    user_cfg = tmp_path / "config.toml"
    user_cfg.write_text("""
[providers.gemini]
safety_block = "BLOCK_EVERYTHING"
""")
    with pytest.raises(ConfigError, match="safety_block"):
        Config.load(user_cfg)


def test_sweep_interval_zero_is_valid() -> None:
    cfg = IndexerConfig(sweep_interval_seconds=0)
    assert cfg.sweep_interval_seconds == 0


def test_sweep_rate_below_minimum_rejected() -> None:
    with pytest.raises(ValidationError):
        IndexerConfig(sweep_rate_sections_per_second=0.0)


def test_search_config_defaults() -> None:
    from contextd.config import SearchConfig

    cfg = SearchConfig()
    assert cfg.mode == "hybrid"
    assert cfg.rrf_k == 60
    assert cfg.fetch_k == 50
    assert cfg.vector_weight == 1.0
    assert cfg.fulltext_weight == 1.0


def test_load_default_has_hybrid_search() -> None:
    cfg = Config.load_default()
    assert cfg.search.mode == "hybrid"
    assert cfg.search.rrf_k == 60
    assert cfg.search.fetch_k == 50


def test_search_config_rejects_both_weights_zero() -> None:
    from contextd.config import SearchConfig

    with pytest.raises(ValidationError, match="must not both be zero"):
        SearchConfig(vector_weight=0.0, fulltext_weight=0.0)


def test_search_config_one_zero_weight_is_valid() -> None:
    from contextd.config import SearchConfig

    cfg = SearchConfig(vector_weight=0.0)
    assert cfg.vector_weight == 0.0
    assert cfg.fulltext_weight == 1.0


def test_search_config_rejects_rrf_k_zero() -> None:
    from contextd.config import SearchConfig

    with pytest.raises(ValidationError):
        SearchConfig(rrf_k=0)


def test_search_config_rejects_unknown_mode(tmp_path: Path) -> None:
    user_cfg = tmp_path / "config.toml"
    user_cfg.write_text("""
[search]
mode = "fuzzy"
""")
    with pytest.raises(ConfigError, match=r"hybrid.*fulltext.*vector|search"):
        Config.load(user_cfg)


def test_search_config_override(tmp_path: Path) -> None:
    user_cfg = tmp_path / "config.toml"
    user_cfg.write_text("""
[search]
mode = "fulltext"
rrf_k = 30
fetch_k = 100
""")
    cfg = Config.load(user_cfg)
    assert cfg.search.mode == "fulltext"
    assert cfg.search.rrf_k == 30
    assert cfg.search.fetch_k == 100
    # Unspecified weights fall back to defaults.
    assert cfg.search.vector_weight == 1.0
