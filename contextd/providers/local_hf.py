"""In-process ``sentence-transformers`` embedder with late-chunking support.

``LocalHFEmbedder`` is the one :class:`TokenEmbedder` implementation. It is
selected with ``providers.embedding = "local_hf"`` and needs the optional
extra ``contextd[late]``; the module itself imports without it and the
dependency is only resolved on first use, so a config that merely mentions
``local_hf`` never breaks an unrelated command.

Late chunking (:meth:`LocalHFEmbedder.embed_spans`):

1. tokenize the whole text once with character offsets and cut it into
   windows of at most ``max_context_tokens`` tokens (overlapping by a
   quarter of the window so no span sits on a hard edge);
2. run one forward pass per window, taking the per-token vectors rather than
   the pooled sentence embedding;
3. for every requested character span, mean-pool the vectors of the tokens
   intersecting it — across windows when the span straddles one — and
   normalise if configured. A span that intersects no token borrows the
   nearest token's vector so it never becomes a zero vector.

Every step above works on plain Python lists so the pooling math is unit
tested with a fake tokenizer / model injected through ``_tokenizer`` /
``_model``; the real model's tensors are converted with ``.tolist()``.
"""

from __future__ import annotations

import bisect
import datetime as dt
import logging
import math
from dataclasses import dataclass
from typing import Any

from contextd.config import LocalHFConfig
from contextd.providers.base import TokenEmbedder, UsageRecord

_log = logging.getLogger(__name__)

# Fraction of the window body carried over into the next window so tokens at
# a window edge are also seen with context on at least one side.
_WINDOW_OVERLAP = 0.25
# Special tokens ([CLS]/[SEP] or model equivalents) reserved out of each
# window when the tokenizer cannot report the count itself.
_DEFAULT_SPECIAL_TOKENS = 2


def _import_sentence_transformers() -> Any:
    """Import the optional ``contextd[late]`` extra or raise a clear error."""
    try:
        import sentence_transformers  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "providers.embedding = 'local_hf' requires the optional dependency: "
            "pip install 'contextd[late]'"
        ) from exc
    return sentence_transformers


def _as_rows(matrix: Any) -> list[list[float]]:
    """Per-token vectors as plain float lists (torch / numpy / list input)."""
    if hasattr(matrix, "detach"):
        matrix = matrix.detach()
    if hasattr(matrix, "cpu"):
        matrix = matrix.cpu()
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    return [[float(x) for x in row] for row in matrix]


def _as_vector(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(x) for x in vector]


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 0.0 else vector


@dataclass
class _Window:
    """One forward-pass window: its text, the absolute char offsets of every
    token the model will see (special tokens carry ``(-1, -1)``) and each
    token's pooling weight — ``1 / number of windows containing that token``
    so a token in an overlap contributes its two views at half weight each
    instead of counting twice (special tokens weigh 0)."""

    text: str
    starts: list[int]
    ends: list[int]
    weights: list[float]

    @property
    def token_count(self) -> int:
        return len(self.starts)


class LocalHFEmbedder(TokenEmbedder):
    def __init__(
        self,
        config: LocalHFConfig,
        *,
        _model: Any | None = None,
        _tokenizer: Any | None = None,
    ) -> None:
        self._cfg = config
        self._model = _model
        self._tokenizer = _tokenizer
        self._last_usage: UsageRecord | None = None

    # -- EmbeddingProvider ---------------------------------------------------

    @property
    def dimensions(self) -> int:
        return self._cfg.dimensions

    @property
    def max_context_tokens(self) -> int:
        return self._cfg.max_context_tokens

    def last_usage(self) -> UsageRecord | None:
        return self._last_usage

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model, tokenizer = self._load()
        # Some models reject empty strings; a single space keeps the
        # one-vector-per-input alignment callers depend on.
        safe = [t if t.strip() else " " for t in texts]
        try:
            raw = model.encode(
                safe,
                batch_size=self._cfg.batch_size,
                normalize_embeddings=self._cfg.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception:
            _log.error("local_hf: model %s failed to embed %d texts", self._cfg.model, len(texts))
            raise
        vectors = [_as_vector(v) for v in raw]
        for vector in vectors:
            self._check_dimensions(len(vector))
        tokens = sum(len(self._encode(tokenizer, t, add_special_tokens=True).ids) for t in safe)
        self._record_usage(tokens)
        return vectors

    # -- TokenEmbedder -------------------------------------------------------

    def embed_spans(self, text: str, spans: list[tuple[int, int]]) -> list[list[float]]:
        if not spans:
            return []
        model, tokenizer = self._load()
        windows = self._windows(tokenizer, text)
        if not windows:
            # Nothing tokenisable (empty / whitespace-only text): there is no
            # token to pool or to borrow, so embed the span texts directly.
            return self.embed([text[s:e] for s, e in spans])

        dims = len(spans)
        sums: list[list[float] | None] = [None] * dims
        counts = [0.0] * dims
        # (window index, token index) of the nearest token for spans that
        # intersect nothing, resolved before the forward pass from offsets.
        borrow: dict[int, tuple[int, int]] = {}
        ranges: list[list[tuple[int, int, int]]] = [[] for _ in windows]  # per window
        for si, (s, e) in enumerate(spans):
            best: tuple[int, int, int] | None = None  # (distance, window, token)
            for wi, w in enumerate(windows):
                a, b = _intersecting(w, s, e)
                if a < b:
                    ranges[wi].append((si, a, b))
                    counts[si] += sum(w.weights[a:b])
                    continue
                cand = _nearest(w, s, e)
                if cand is not None and (best is None or cand[0] < best[0]):
                    best = (cand[0], wi, cand[1])
            if counts[si] == 0 and best is not None:
                borrow[si] = (best[1], best[2])

        try:
            matrices = model.encode(
                [w.text for w in windows],
                batch_size=self._cfg.batch_size,
                output_value="token_embeddings",
                convert_to_numpy=False,
                convert_to_tensor=False,
                show_progress_bar=False,
            )
        except Exception:
            _log.error(
                "local_hf: model %s failed on %d window(s) of %d chars",
                self._cfg.model,
                len(windows),
                len(text),
            )
            raise

        out: list[list[float] | None] = [None] * dims
        for wi, (w, matrix) in enumerate(zip(windows, matrices, strict=True)):
            rows = _as_rows(matrix)
            if len(rows) != w.token_count:
                raise RuntimeError(
                    f"local_hf: model {self._cfg.model!r} returned {len(rows)} token "
                    f"vectors for a window of {w.token_count} tokens; token/offset "
                    "alignment is broken, refusing to pool"
                )
            if rows:
                self._check_dimensions(len(rows[0]))
            for si, a, b in ranges[wi]:
                acc = sums[si]
                for row, wgt in zip(rows[a:b], w.weights[a:b], strict=True):
                    if acc is None:
                        acc = [wgt * y for y in row]
                    else:
                        acc = [x + wgt * y for x, y in zip(acc, row, strict=True)]
                sums[si] = acc
            for si, (bw, bt) in borrow.items():
                if bw == wi:
                    out[si] = rows[bt]

        vectors: list[list[float]] = []
        for si in range(dims):
            acc = sums[si]
            if acc is not None:
                vec = [x / counts[si] for x in acc]
            else:
                borrowed = out[si]
                if borrowed is None:  # unreachable: every span pools or borrows
                    raise RuntimeError(f"local_hf: no token vector for span {spans[si]}")
                vec = borrowed
            vectors.append(_normalise(vec) if self._cfg.normalize else vec)
        self._record_usage(sum(w.token_count for w in windows))
        return vectors

    # -- internals -----------------------------------------------------------

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            st = _import_sentence_transformers()
            try:
                self._model = st.SentenceTransformer(self._cfg.model, device=self._cfg.device)
                self._model.max_seq_length = self._cfg.max_context_tokens
            except Exception:
                _log.error(
                    "local_hf: failed to load model %s on %s", self._cfg.model, self._cfg.device
                )
                raise
        if self._tokenizer is None:
            self._tokenizer = self._model.tokenizer
        return self._model, self._tokenizer

    def _check_dimensions(self, got: int) -> None:
        if got != self._cfg.dimensions:
            raise ValueError(
                f"Embedding model {self._cfg.model!r} returned a {got}-dimension vector "
                f"but the configured / indexed dimension is {self._cfg.dimensions}. Either "
                f"choose a model that emits {self._cfg.dimensions}-dim vectors (e.g. "
                "BAAI/bge-m3), or update providers.local_hf.dimensions together with the "
                "vector-index DDL in the migrations."
            )

    def _record_usage(self, tokens: int) -> None:
        self._last_usage = UsageRecord(
            provider="local_hf",
            model=self._cfg.model,
            call_site="embedding",
            input_tokens=tokens,
            output_tokens=0,
            timestamp=dt.datetime.now(dt.UTC).isoformat(),
        )

    @staticmethod
    def _encode(
        tokenizer: Any, text: str, *, add_special_tokens: bool, max_length: int | None = None
    ) -> _Encoding:
        kwargs: dict[str, Any] = {
            "add_special_tokens": add_special_tokens,
            "return_offsets_mapping": True,
        }
        if max_length is None:
            kwargs["truncation"] = False
        else:
            kwargs["truncation"] = True
            kwargs["max_length"] = max_length
        enc = tokenizer(text, **kwargs)
        ids = [int(i) for i in enc["input_ids"]]
        offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
        return _Encoding(ids, offsets)

    def _windows(self, tokenizer: Any, text: str) -> list[_Window]:
        """Cut ``text`` into token windows of at most ``max_context_tokens``
        (special tokens included) and tokenize each window as the model will
        see it, recording absolute character offsets per token."""
        full = self._encode(tokenizer, text, add_special_tokens=False)
        body = [(s, e) for s, e in full.offsets if e > s]
        if not body:
            return []
        special = _DEFAULT_SPECIAL_TOKENS
        count_special = getattr(tokenizer, "num_special_tokens_to_add", None)
        if callable(count_special):
            try:
                special = int(count_special())
            except Exception:  # pragma: no cover - tokenizer quirk, keep the default
                special = _DEFAULT_SPECIAL_TOKENS
        width = max(1, self.max_context_tokens - special)
        stride = max(1, width - int(width * _WINDOW_OVERLAP))
        windows: list[_Window] = []
        i = 0
        while True:
            j = min(i + width, len(body))
            base = body[i][0]
            chunk_text = text[base : body[j - 1][1]]
            enc = self._encode(
                tokenizer,
                chunk_text,
                add_special_tokens=True,
                max_length=self.max_context_tokens,
            )
            starts = [base + s if e > s else -1 for s, e in enc.offsets]
            ends = [base + e if e > s else -1 for s, e in enc.offsets]
            windows.append(_Window(chunk_text, starts, ends, []))
            if j >= len(body):
                break
            i += stride
        multiplicity: dict[tuple[int, int], int] = {}
        for w in windows:
            for s, e in zip(w.starts, w.ends, strict=True):
                if s >= 0:
                    multiplicity[(s, e)] = multiplicity.get((s, e), 0) + 1
        for w in windows:
            w.weights = [
                1.0 / multiplicity[(s, e)] if s >= 0 else 0.0
                for s, e in zip(w.starts, w.ends, strict=True)
            ]
        return windows


@dataclass(frozen=True)
class _Encoding:
    ids: list[int]
    offsets: list[tuple[int, int]]


def _real_tokens(w: _Window) -> tuple[list[int], list[int], list[int]]:
    """Indices, starts and ends of the window's non-special tokens."""
    idx = [i for i, s in enumerate(w.starts) if s >= 0]
    return idx, [w.starts[i] for i in idx], [w.ends[i] for i in idx]


def _intersecting(w: _Window, s: int, e: int) -> tuple[int, int]:
    """Half-open token index range of ``w`` intersecting chars ``[s, e)``.

    Tokens are in text order with non-decreasing starts and ends, so the
    intersecting set is contiguous: from the first token ending after ``s``
    to the last token starting before ``e``. Special tokens never intersect.
    """
    if e <= s:
        return 0, 0
    idx, starts, ends = _real_tokens(w)
    lo = bisect.bisect_right(ends, s)  # first token with end > s
    hi = bisect.bisect_left(starts, e)  # first token with start >= e
    if lo >= hi:
        return 0, 0
    return idx[lo], idx[hi - 1] + 1


def _nearest(w: _Window, s: int, e: int) -> tuple[int, int] | None:
    """``(distance, token index)`` of the window token closest to ``[s, e)``."""
    idx, starts, ends = _real_tokens(w)
    if not idx:
        return None
    mid = (s + e) / 2.0
    pos = bisect.bisect_left(starts, mid)
    best: tuple[float, int] | None = None
    for k in (pos - 1, pos):
        if 0 <= k < len(idx):
            d = 0.0 if starts[k] <= mid < ends[k] else min(abs(starts[k] - mid), abs(ends[k] - mid))
            if best is None or d < best[0]:
                best = (d, idx[k])
    return None if best is None else (math.ceil(best[0]), best[1])
