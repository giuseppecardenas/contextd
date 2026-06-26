import pytest

from contextd.config import Config
from contextd.providers.factory import (
    ProviderFactoryError,
    build_embedding_provider,
    build_inference_provider,
)
from contextd.providers.gemini import GeminiProvider
from contextd.providers.openai_compat import OpenAICompatProvider
from contextd.providers.router import RoutingInferenceProvider


def test_build_gemini_from_default_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg = Config.load_default()
    provider = build_inference_provider(cfg)
    from contextd.providers.base import InferenceProvider

    assert isinstance(provider, InferenceProvider)


def test_factory_returns_routing_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg = Config.load_default()
    provider = build_inference_provider(cfg)
    assert isinstance(provider, RoutingInferenceProvider)


def test_factory_reuses_provider_instance_when_same_backend_named_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three call-sites all set to gemini → single GeminiProvider instance
    shared across all three router slots."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg = Config.load_default()
    provider = build_inference_provider(cfg)
    assert isinstance(provider, RoutingInferenceProvider)
    summary = provider._slots["summary"]
    inference = provider._slots["inference"]
    translation = provider._slots["translation"]
    assert summary is inference is translation


def test_factory_picks_openai_compat_for_summary_when_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    user_cfg = tmp_path / "config.toml"  # type: ignore[operator]
    user_cfg.write_text("""
[providers]
summary = "openai_compat"
inference = "gemini"
translation = "gemini"
""")
    cfg = Config.load(user_cfg)
    provider = build_inference_provider(cfg)
    assert isinstance(provider, RoutingInferenceProvider)
    assert isinstance(provider._slots["summary"], OpenAICompatProvider)
    assert isinstance(provider._slots["inference"], GeminiProvider)
    assert isinstance(provider._slots["translation"], GeminiProvider)
    # Inference + translation share the same Gemini instance.
    assert provider._slots["inference"] is provider._slots["translation"]


def test_factory_raises_when_gemini_call_site_used_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = Config.load_default()
    with pytest.raises(ProviderFactoryError, match="GEMINI_API_KEY"):
        build_inference_provider(cfg)


def test_factory_raises_when_openai_compat_api_key_env_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    monkeypatch.delenv("FAKE_OPENAI_KEY", raising=False)
    user_cfg = tmp_path / "config.toml"  # type: ignore[operator]
    user_cfg.write_text("""
[providers]
summary = "openai_compat"
inference = "openai_compat"
translation = "openai_compat"

[providers.openai_compat]
api_key_env = "FAKE_OPENAI_KEY"
""")
    cfg = Config.load(user_cfg)
    with pytest.raises(ProviderFactoryError, match="FAKE_OPENAI_KEY"):
        build_inference_provider(cfg)


def test_factory_skips_gemini_construction_when_no_call_site_uses_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    """If all three call-sites use openai_compat, GEMINI_API_KEY is irrelevant."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    user_cfg = tmp_path / "config.toml"  # type: ignore[operator]
    user_cfg.write_text("""
[providers]
summary = "openai_compat"
inference = "openai_compat"
translation = "openai_compat"
""")
    cfg = Config.load(user_cfg)
    provider = build_inference_provider(cfg)
    assert isinstance(provider, RoutingInferenceProvider)
    assert isinstance(provider._slots["summary"], OpenAICompatProvider)


def test_build_voyage_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    cfg = Config.load_default()
    provider = build_embedding_provider(cfg)
    from contextd.providers.base import EmbeddingProvider

    assert isinstance(provider, EmbeddingProvider)


def test_build_openai_compat_embedding_keyless(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    """A local embedding server with no api_key_env builds without any key."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    user_cfg = tmp_path / "config.toml"  # type: ignore[operator]
    user_cfg.write_text("""
[providers]
embedding = "openai_compat"
""")
    cfg = Config.load(user_cfg)
    from contextd.providers.openai_compat_embedding import OpenAICompatEmbeddingProvider

    provider = build_embedding_provider(cfg)
    assert isinstance(provider, OpenAICompatEmbeddingProvider)


def test_build_openai_compat_embedding_raises_when_api_key_env_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    monkeypatch.delenv("FAKE_EMBED_KEY", raising=False)
    user_cfg = tmp_path / "config.toml"  # type: ignore[operator]
    user_cfg.write_text("""
[providers]
embedding = "openai_compat"

[providers.openai_compat_embedding]
api_key_env = "FAKE_EMBED_KEY"
""")
    cfg = Config.load(user_cfg)
    with pytest.raises(ProviderFactoryError, match="FAKE_EMBED_KEY"):
        build_embedding_provider(cfg)
