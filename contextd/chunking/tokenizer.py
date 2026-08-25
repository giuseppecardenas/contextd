"""Tokenizer abstraction for size-bounded chunking.

Strategies never decode tokens back to text — they work on *character
offsets* per token, so a chunk boundary chosen in token space maps exactly
onto the source text (and therefore onto file line numbers). That is why the
protocol exposes ``offsets`` rather than ``encode``/``decode``.

Three implementations:

* :class:`VoyageTokenizer` — the HuggingFace ``tokenizers`` model published
  by Voyage for each embedding model (``voyageai/<model>`` on the Hub; the
  same lookup ``voyageai.Client.count_tokens`` performs). The Hub fetch can
  fail on an offline or rate-limited machine, so construction degrades to
  :class:`WordTokenizer` with one warning instead of aborting a bootstrap —
  the same policy as ``VoyageProvider._count_batch_tokens``.
* :class:`TiktokenTokenizer` — ``o200k_base`` by default, for OpenAI-compatible
  embedders. Optional extra ``contextd[tiktoken]``.
* :class:`WordTokenizer` — whitespace words scaled by 1.3; the offline and
  unit-test tokenizer.

``resolve_tokenizer`` is the single selection point from config.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

from contextd.corpus_config import TokenizerName

_log = logging.getLogger(__name__)

# Empirical prose ratio: ~1.3 subword tokens per whitespace word for the
# tokenizers contextd targets. Applied as a multiplier on the word count so
# the word tokenizer under-packs rather than over-packs a chunk.
_WORDS_TO_TOKENS = 1.3

_WORD = re.compile(r"\S+")


class Tokenizer(Protocol):
    """Token counting plus per-token character offsets."""

    @property
    def id(self) -> str:
        """Stable identifier that participates in the chunk fingerprint."""
        ...

    def count(self, text: str) -> int: ...

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """``(start, end)`` character offsets of every token, in order.

        Offsets are half-open and non-decreasing; whitespace between tokens
        need not be covered. An empty text yields an empty list.
        """
        ...


class WordTokenizer:
    """Whitespace-word tokenizer with a fixed word→token scale."""

    id = "words"

    def count(self, text: str) -> int:
        words = len(_WORD.findall(text))
        return int(words * _WORDS_TO_TOKENS + 0.5) if words else 0

    def offsets(self, text: str) -> list[tuple[int, int]]:
        # One "token" per word: the scale is applied by ``count`` only, so
        # offsets stay aligned with real word boundaries (a chunk of N words
        # is reported as ~1.3N tokens, which is the conservative direction).
        return [(m.start(), m.end()) for m in _WORD.finditer(text)]


class VoyageTokenizer:
    """HuggingFace ``tokenizers`` model for a Voyage embedding model."""

    def __init__(self, model: str, *, hf_tokenizer: Any | None = None) -> None:
        self._model = model
        self._tok = hf_tokenizer if hf_tokenizer is not None else _load_hf(f"voyageai/{model}")

    @property
    def id(self) -> str:
        return f"voyage:{self._model}"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        enc = self._tok.encode(text, add_special_tokens=False)
        return [(int(s), int(e)) for s, e in enc.offsets]


class TiktokenTokenizer:
    """``tiktoken`` encoding (default ``o200k_base``)."""

    def __init__(self, encoding: str = "o200k_base", *, enc: Any | None = None) -> None:
        self._name = encoding
        if enc is None:
            enc = _import_tiktoken().get_encoding(encoding)
        self._enc = enc

    @property
    def id(self) -> str:
        return f"tiktoken:{self._name}"

    def count(self, text: str) -> int:
        return len(self._enc.encode(text)) if text else 0

    def offsets(self, text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        ids = self._enc.encode(text)
        _, starts = self._enc.decode_with_offsets(ids)
        # decode_with_offsets yields each token's start; ends are the next start.
        out: list[tuple[int, int]] = []
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else len(text)
            out.append((int(s), int(e)))
        return out


def _import_tiktoken() -> Any:
    """Import the optional ``tiktoken`` extra or raise a config-time error."""
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "tokenizer = 'tiktoken' requires the optional dependency: "
            "pip install 'contextd[tiktoken]'"
        ) from exc
    return tiktoken


def _load_hf(name: str) -> Any:
    from tokenizers import Tokenizer as HFTokenizer

    return HFTokenizer.from_pretrained(name)


def resolve_tokenizer(
    name: TokenizerName,
    *,
    embedding_provider: str,
    voyage_model: str,
) -> Tokenizer:
    """Pick the tokenizer for a corpus.

    ``auto`` follows the embedder: Voyage's tokenizer for ``voyage``, tiktoken
    when installed for anything else, words otherwise. Explicit ``voyage`` /
    ``tiktoken`` requests degrade to ``words`` with a warning when their
    backing model cannot be loaded — chunk sizes then drift by the
    word/token ratio, which is a tolerable outcome; aborting a bootstrap is not.
    """
    if name == "auto":
        if embedding_provider == "voyage":
            name = "voyage"
        else:
            try:
                _import_tiktoken()
            except RuntimeError:
                name = "words"
            else:
                name = "tiktoken"
    if name == "voyage":
        try:
            return VoyageTokenizer(voyage_model)
        except Exception:
            _log.warning(
                "chunking: Voyage tokenizer for %s unavailable; using the word tokenizer",
                voyage_model,
                exc_info=True,
            )
            return WordTokenizer()
    if name == "tiktoken":
        try:
            return TiktokenTokenizer()
        except Exception:
            _log.warning("chunking: tiktoken unavailable; using the word tokenizer", exc_info=True)
            return WordTokenizer()
    return WordTokenizer()
