"""Factory functions that construct concrete providers from config + env."""

from __future__ import annotations

import os

from contextd.config import Config
from contextd.providers.base import EmbeddingProvider, InferenceProvider
from contextd.providers.gemini import GeminiProvider
from contextd.providers.voyage import VoyageProvider


class ProviderFactoryError(RuntimeError):
    """Raised when a provider cannot be constructed (missing env var, etc)."""


def build_inference_provider(cfg: Config) -> InferenceProvider:
    name = cfg.providers.inference
    if name == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ProviderFactoryError(
                "GEMINI_API_KEY is required when providers.inference = 'gemini'. "
                "Get a key at https://aistudio.google.com/app/apikey"
            )
        return GeminiProvider(cfg.providers.gemini, api_key=key)
    raise ProviderFactoryError(f"Unknown inference provider: {name!r}")


def build_embedding_provider(cfg: Config) -> EmbeddingProvider:
    name = cfg.providers.embedding
    if name == "voyage":
        key = os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise ProviderFactoryError(
                "VOYAGE_API_KEY is required when providers.embedding = 'voyage'. "
                "Get a key at https://www.voyageai.com/"
            )
        return VoyageProvider(cfg.providers.voyage, api_key=key)
    raise ProviderFactoryError(f"Unknown embedding provider: {name!r}")
