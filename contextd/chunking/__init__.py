"""Retrieval chunking: pure, synchronous text → chunk logic.

Everything under this package is free of storage and provider imports except
through the injected protocols (``Tokenizer``, ``EmbeddingProvider``,
``InferenceProvider``, ``TokenEmbedder``); the indexer phases own the I/O.
"""

from contextd.chunking.fingerprint import config_fingerprint, unit_fingerprint

__all__ = ["config_fingerprint", "unit_fingerprint"]
