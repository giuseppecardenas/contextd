"""Strategy registry: profile → chunker.

``build_chunker`` is the one place a strategy name is mapped to a class and
its dependencies are checked. Strategies that need a provider (``semantic``,
``late``, ``propositions``) or an optional extra (``code``) raise
:class:`ChunkingConfigError` here — at pipeline construction — rather than
part-way through a bootstrap.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from contextd.chunking.strategies.base import ChunkStrategy
from contextd.chunking.strategies.recursive import RecursiveStrategy
from contextd.chunking.strategies.sentence_window import SentenceWindowStrategy
from contextd.chunking.strategies.structural import StructuralStrategy
from contextd.chunking.strategies.window import WindowStrategy
from contextd.chunking.tokenizer import Tokenizer
from contextd.corpus_config import ChunkProfile

if TYPE_CHECKING:
    from contextd.inference.prompts import PromptRenderer
    from contextd.providers.base import EmbeddingProvider, InferenceProvider


class ChunkingConfigError(ValueError):
    """A profile names a strategy whose dependencies are not available."""


@dataclass
class StrategyDeps:
    """Optional collaborators a strategy may require."""

    embedder: EmbeddingProvider | None = None
    inference: InferenceProvider | None = None
    renderer: PromptRenderer | None = None
    token_embedder: object | None = None  # TokenEmbedder; typed loosely until WS9 lands


_Factory = Callable[[ChunkProfile, Tokenizer, StrategyDeps], ChunkStrategy]


def _require(profile: ChunkProfile, value: object | None, what: str) -> None:
    if value is None:
        raise ChunkingConfigError(
            f"profile {profile.name!r}: strategy {profile.strategy!r} requires {what}"
        )


def _structural(_p: ChunkProfile, tok: Tokenizer, _d: StrategyDeps) -> ChunkStrategy:
    return StructuralStrategy(tok)


def _window(_p: ChunkProfile, tok: Tokenizer, _d: StrategyDeps) -> ChunkStrategy:
    return WindowStrategy(tok)


def _recursive(_p: ChunkProfile, tok: Tokenizer, _d: StrategyDeps) -> ChunkStrategy:
    return RecursiveStrategy(tok)


def _sentence_window(_p: ChunkProfile, tok: Tokenizer, _d: StrategyDeps) -> ChunkStrategy:
    return SentenceWindowStrategy(tok)


def _semantic(p: ChunkProfile, tok: Tokenizer, d: StrategyDeps) -> ChunkStrategy:
    from contextd.chunking.strategies.semantic import SemanticStrategy

    _require(p, d.embedder, "an embedding provider (providers.embedding)")
    assert d.embedder is not None
    return SemanticStrategy(tok, d.embedder)


def _propositions(p: ChunkProfile, tok: Tokenizer, d: StrategyDeps) -> ChunkStrategy:
    from contextd.chunking.strategies.propositions import PropositionsStrategy

    _require(p, d.inference, "an inference provider (providers.summary)")
    _require(p, d.renderer, "a prompt renderer")
    assert d.inference is not None and d.renderer is not None
    return PropositionsStrategy(tok, d.inference, d.renderer)


def _code(p: ChunkProfile, tok: Tokenizer, d: StrategyDeps) -> ChunkStrategy:
    from contextd.chunking.strategies.code import make_code

    return make_code(p, tok, d)


STRATEGY_REGISTRY: dict[str, _Factory] = {
    "structural": _structural,
    "window": _window,
    "recursive": _recursive,
    "sentence_window": _sentence_window,
    "semantic": _semantic,
    "propositions": _propositions,
    "code": _code,
}


def register(name: str, factory: _Factory) -> None:
    STRATEGY_REGISTRY[name] = factory


def build_chunker(
    profile: ChunkProfile, tokenizer: Tokenizer, deps: StrategyDeps | None = None
) -> ChunkStrategy:
    try:
        factory = STRATEGY_REGISTRY[profile.strategy]
    except KeyError as exc:
        raise ChunkingConfigError(
            f"profile {profile.name!r}: strategy {profile.strategy!r} is not available "
            f"(registered: {sorted(STRATEGY_REGISTRY)})"
        ) from exc
    return factory(profile, tokenizer, deps or StrategyDeps())


__all__ = [
    "STRATEGY_REGISTRY",
    "ChunkStrategy",
    "ChunkingConfigError",
    "StrategyDeps",
    "build_chunker",
    "register",
]
