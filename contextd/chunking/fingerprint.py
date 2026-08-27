"""Chunk-configuration fingerprints.

Chunks are derived data: a parent unit's chunks are a pure function of
(chunking config, tokenizer, parent content). The indexer stores
``unit_fingerprint`` on every Section/File whose chunks are committed and
skips the unit when the stored value equals the current one. That single
string compare is what gives resume-after-crash, incremental re-index, the
daemon sweep, and "config changed" all the same semantics — no per-chunk hash
diffing is needed.

``config_fingerprint`` (the corpus-wide part) is also persisted on the
``Corpus`` node so ``contextd status`` can show it and the daemon can detect
drift at startup.
"""

from __future__ import annotations

import hashlib
import json

from contextd.corpus_config import ChunkingSection

# Bump when chunk *semantics* change without a config change — e.g. a
# strategy's packing rule is fixed — so existing graphs re-chunk once.
CHUNKING_ALGORITHM_VERSION = 1


def config_fingerprint(chunking: ChunkingSection, tokenizer_id: str) -> str:
    """Stable hash of everything that shapes chunk output except the content.

    A profile's ``weight`` is a query-time RRF knob — it changes how chunks
    are *ranked*, never which chunks exist or what they contain — so it is
    dropped from the payload. Hashing it would turn every weight tuning into
    a full re-chunk of the corpus (the runeledger corpus is ~2.2M embedding
    tokens), which is exactly the cost the fingerprint gate exists to avoid.
    """
    config = chunking.model_dump(mode="json")
    for profile in config.get("profiles", []):
        profile.pop("weight", None)
    payload = {
        "algorithm": CHUNKING_ALGORITHM_VERSION,
        "tokenizer": tokenizer_id,
        "config": config,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def unit_fingerprint(config_fp: str, parent_hash: str) -> str:
    """Per-parent fingerprint: the config fingerprint bound to the parent's content hash."""
    return hashlib.sha256(f"{config_fp}:{parent_hash}".encode()).hexdigest()
