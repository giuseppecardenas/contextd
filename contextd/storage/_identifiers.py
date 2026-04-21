"""Backend input-validation helpers.

Backend methods accept ``label``, ``edge_type``, and ``property_name`` as
``str`` and splice them into Cypher because neither Memgraph nor Neo4j support
parameterising identifier positions (labels, relationship types, property
names in CREATE/SET patterns). The upstream callers — the indexer and the
ontology-validation layer — already restrict these to known values, but
defence-in-depth at the backend boundary closes the window where:

- AI-inferred edges land with LLM-generated property keys (M5+).
- The MCP ``query_graph`` tool forwards tool arguments into search (M7+).

The safe subset is a Python identifier shape: ``[A-Za-z_][A-Za-z0-9_]*``.
Any call site that cannot accept an identifier (e.g., a whole property
dict) must parameterise via ``$bind`` instead of splicing.

The module also hosts numeric-input guards (``validate_search_k``,
``validate_threshold``) that both backends share — silent coercions
like ``k=True -> 1`` or ``k=9.9 -> 9`` would otherwise succeed.
"""

from __future__ import annotations

import math
import re
from typing import Final

_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(value: str, *, kind: str) -> str:
    """Return ``value`` unchanged if it matches ``[A-Za-z_][A-Za-z0-9_]*``.

    ``kind`` is echoed in the error message so callers know which argument
    failed (``label``, ``edge_type``, ``property_name``).
    """
    if not isinstance(value, str) or not _IDENTIFIER_RE.match(value):
        raise ValueError(
            f"{kind} must match [A-Za-z_][A-Za-z0-9_]* to be safe for Cypher "
            f"interpolation; got {value!r}"
        )
    return value


def validate_property_keys(properties: dict[str, object], *, context: str) -> None:
    """Raise ``ValueError`` if any key in ``properties`` is not a safe identifier.

    ``context`` names the caller (e.g., ``upsert_node(label='File')``) so the
    error points at the failing call site, not just the offending key.
    """
    for key in properties:
        if not isinstance(key, str) or not _IDENTIFIER_RE.match(key):
            raise ValueError(
                f"{context}: property key {key!r} is not a safe Cypher identifier "
                f"([A-Za-z_][A-Za-z0-9_]*). Rename the property or route it "
                f"through a parameterised bind instead of identifier interpolation."
            )


def validate_search_k(k: int) -> int:
    """Return ``k`` unchanged if it is a positive non-bool ``int``.

    ``isinstance(True, int)`` is ``True`` in Python — the bool check must
    precede the int check. Both backends silently mis-interpret ``True``
    as ``1`` or ``9.9`` as ``9`` if we accept anything beyond ``int``.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError(f"k must be a non-bool int; got {k!r} ({type(k).__name__})")
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k!r}")
    return k


def validate_threshold(threshold: float | None) -> float | None:
    """Return ``threshold`` unchanged, or raise if it is non-finite or out of [0, 1].

    Both backends treat threshold as a cosine-similarity floor in ``[0, 1]``.
    ``nan`` / ``inf`` would bypass the parameter bind on backends that
    f-string the value; values outside ``[0, 1]`` yield empty-set or
    nonsense results depending on the backend's scoring origin.
    """
    if threshold is None:
        return None
    if not isinstance(threshold, int | float) or isinstance(threshold, bool):
        raise ValueError(
            f"threshold must be a finite float in [0, 1]; got {threshold!r} "
            f"({type(threshold).__name__})"
        )
    if not math.isfinite(threshold):
        raise ValueError(f"threshold must be finite; got {threshold!r}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1]; got {threshold!r}")
    return float(threshold)
