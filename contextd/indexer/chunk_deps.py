"""Chunking collaborators assembled once per pipeline invocation.

``build_chunking_deps`` is the single place the corpus ``[chunking]`` config
is turned into runnable strategies: it resolves the tokenizer, builds one
chunker per profile (raising ``ChunkingConfigError`` at construction for a
profile whose provider or extra is missing — never mid-bootstrap), and
computes the corpus-wide config fingerprint the phases gate on.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextd.chunking.fingerprint import config_fingerprint
from contextd.chunking.strategies import ChunkStrategy, StrategyDeps, build_chunker
from contextd.chunking.tokenizer import Tokenizer, resolve_tokenizer
from contextd.config import Config
from contextd.corpus_config import ChunkingSection, CorpusConfig
from contextd.inference.prompts import PromptRenderer
from contextd.providers.base import EmbeddingProvider, InferenceProvider


@dataclass
class ChunkingDeps:
    config: ChunkingSection
    tokenizer: Tokenizer
    embedder: EmbeddingProvider | None
    """``None`` only for offline estimates (``validate=False``)."""
    config_fp: str
    inference: InferenceProvider | None = None
    """Provider for ``prefix = "llm"`` and ``questions`` augmentation."""
    renderer: PromptRenderer | None = None
    _chunkers: dict[tuple[str, str], ChunkStrategy] | None = None

    def chunker(self, suffix: str, profile_name: str) -> ChunkStrategy:
        """Strategy for ``(suffix, profile)``, built once and cached.

        Suffix overrides may change a profile's strategy per file type, so the
        cache key includes the suffix.
        """
        if self._chunkers is None:
            self._chunkers = {}
        key = (suffix, profile_name)
        cached = self._chunkers.get(key)
        if cached is not None:
            return cached
        profile = next(p for p in self.config.profiles_for(suffix) if p.name == profile_name)
        strategy = build_chunker(
            profile,
            self.tokenizer,
            StrategyDeps(embedder=self.embedder, inference=self.inference, renderer=self.renderer),
        )
        self._chunkers[key] = strategy
        return strategy

    def validate(self, suffixes: set[str]) -> None:
        """Build every (suffix, profile) chunker up front so config errors
        surface before any graph write or provider call."""
        for suffix in suffixes or {".md"}:
            for p in self.config.profiles_for(suffix):
                self.chunker(suffix, p.name)


def build_chunking_deps(
    cfg: Config,
    corpus_cfg: CorpusConfig,
    *,
    embedder: EmbeddingProvider | None,
    inference: InferenceProvider | None,
    renderer: PromptRenderer | None,
    validate: bool = True,
) -> ChunkingDeps | None:
    """``None`` when the corpus has chunking disabled.

    ``validate=False`` (used by ``--estimate-only``) skips building the
    provider-backed strategies so an estimate never needs credentials; the
    estimator substitutes ``structural`` for them.
    """
    chunking = corpus_cfg.chunking
    if not chunking.enabled:
        return None
    tokenizer = resolve_tokenizer(
        chunking.tokenizer,
        embedding_provider=cfg.providers.embedding,
        voyage_model=corpus_cfg.embedding.model or cfg.providers.voyage.model,
    )
    deps = ChunkingDeps(
        config=chunking,
        tokenizer=tokenizer,
        embedder=embedder,
        config_fp=config_fingerprint(chunking, tokenizer.id),
        inference=inference,
        renderer=renderer,
    )
    if validate:
        deps.validate({".md", *chunking.suffix_overrides})
    return deps
