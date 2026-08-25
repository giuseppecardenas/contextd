"""Factory functions that construct concrete providers from config + env."""

from __future__ import annotations

import os

from contextd.config import Config, openai_compat_profile
from contextd.providers.base import EmbeddingProvider, InferenceProvider
from contextd.providers.gemini import GeminiProvider
from contextd.providers.local_hf import LocalHFEmbedder
from contextd.providers.openai_compat import OpenAICompatProvider
from contextd.providers.openai_compat_embedding import OpenAICompatEmbeddingProvider
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
    cache: dict[str, InferenceProvider] = {}

    def _get(ref: str) -> InferenceProvider:
        if ref in cache:
            return cache[ref]
        profile = openai_compat_profile(ref)
        if ref == "gemini":
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ProviderFactoryError(
                    "GEMINI_API_KEY is required when a provider call-site = 'gemini'. "
                    "Get a key at https://aistudio.google.com/app/apikey"
                )
            inst: InferenceProvider = GeminiProvider(pcfg.gemini, api_key=key)
        elif profile is not None:
            ocfg = pcfg.openai_compat.get(profile)
            if ocfg is None:  # defensive; ProvidersConfig validation normally catches this
                raise ProviderFactoryError(
                    f"No [providers.openai_compat.{profile}] profile is configured"
                )
            api_key: str | None = None
            if ocfg.api_key_env:
                api_key = os.environ.get(ocfg.api_key_env)
                if not api_key:
                    raise ProviderFactoryError(
                        f"providers.openai_compat.{profile}.api_key_env = "
                        f"{ocfg.api_key_env!r} but that env var is unset. Either "
                        "export it or remove api_key_env to run against a keyless "
                        "local server."
                    )
            inst = OpenAICompatProvider(ocfg, api_key=api_key, provider_label=ref)
        else:
            raise ProviderFactoryError(f"Unknown inference provider: {ref!r}")
        cache[ref] = inst
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
    if name == "openai_compat":
        ecfg = cfg.providers.openai_compat_embedding
        api_key: str | None = None
        if ecfg.api_key_env:
            api_key = os.environ.get(ecfg.api_key_env)
            if not api_key:
                raise ProviderFactoryError(
                    f"providers.openai_compat_embedding.api_key_env = "
                    f"{ecfg.api_key_env!r} but that env var is unset. Either export "
                    "it or remove api_key_env to run against a keyless local server."
                )
        return OpenAICompatEmbeddingProvider(ecfg, api_key=api_key)
    if name == "local_hf":
        # No key and no network at construction: the model loads lazily on
        # the first embed (raising a clear RuntimeError when contextd[late]
        # is not installed).
        return LocalHFEmbedder(cfg.providers.local_hf)
    raise ProviderFactoryError(f"Unknown embedding provider: {name!r}")
