import pytest

from contextd.config import Config
from contextd.providers.factory import (
    ProviderFactoryError,
    build_embedding_provider,
    build_inference_provider,
)


def test_build_gemini_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg = Config.load_default()
    provider = build_inference_provider(cfg)
    from contextd.providers.base import InferenceProvider

    assert isinstance(provider, InferenceProvider)


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = Config.load_default()
    with pytest.raises(ProviderFactoryError, match="GEMINI_API_KEY"):
        build_inference_provider(cfg)


def test_build_voyage_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    cfg = Config.load_default()
    provider = build_embedding_provider(cfg)
    from contextd.providers.base import EmbeddingProvider

    assert isinstance(provider, EmbeddingProvider)
