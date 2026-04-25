"""Factory functions that construct concrete providers from config + env."""

from __future__ import annotations

import os

from contextd.config import Config, InferenceProviderName
from contextd.providers.base import EmbeddingProvider, InferenceProvider
from contextd.providers.gemini import GeminiProvider
from contextd.providers.openai_compat import OpenAICompatProvider
from contextd.providers.router import RoutingInferenceProvider
from contextd.providers.voyage import VoyageProvider


class ProviderFactoryError(RuntimeError):
    """Raised when a provider cannot be constructed (missing env var, etc)."""


def build_inference_provider(cfg: Config) -> InferenceProvider:
    """Build a RoutingInferenceProvider with one concrete provider per call-site.

    `summary`, `inference`, and `translation` are independently configured;
    when two or three resolve to the same backend, a single concrete
    provider instance is reused across slots so retry state and HTTP
    clients are shared.
    """
    pcfg = cfg.providers
    cache: dict[InferenceProviderName, InferenceProvider] = {}

    def _get(name: InferenceProviderName) -> InferenceProvider:
        if name in cache:
            return cache[name]
        if name == "gemini":
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ProviderFactoryError(
                    "GEMINI_API_KEY is required when a provider call-site = 'gemini'. "
                    "Get a key at https://aistudio.google.com/app/apikey"
                )
            inst: InferenceProvider = GeminiProvider(pcfg.gemini, api_key=key)
        elif name == "openai_compat":
            api_key: str | None = None
            if pcfg.openai_compat.api_key_env:
                api_key = os.environ.get(pcfg.openai_compat.api_key_env)
                if not api_key:
                    raise ProviderFactoryError(
                        f"providers.openai_compat.api_key_env = "
                        f"{pcfg.openai_compat.api_key_env!r} but that env var is "
                        "unset. Either export it or remove api_key_env to run "
                        "against a keyless local server."
                    )
            inst = OpenAICompatProvider(pcfg.openai_compat, api_key=api_key)
        else:
            raise ProviderFactoryError(f"Unknown inference provider: {name!r}")
        cache[name] = inst
        return inst

    return RoutingInferenceProvider(
        summary=_get(pcfg.summary),
        inference=_get(pcfg.inference),
        translation=_get(pcfg.translation),
    )


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
