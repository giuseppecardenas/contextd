"""Load per-corpus ontology override files and apply them onto a base Ontology.

Override files are JSON objects residing alongside the corpus TOML.  Currently
only ``edge_label_aliases`` is meaningful; the shape is forward-compatible with
additional keys.

Typical call site (inside ``_build_pipeline_deps``)::

    from contextd.ontology.overrides import apply_overrides
    ontology = apply_overrides(ontology, abs_overrides_path)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contextd.ontology.schema import Ontology


class OntologyOverridesError(ValueError):
    """Raised when an overrides file cannot be loaded or is malformed."""


def apply_overrides(
    ontology: Ontology,
    overrides_path: Path,
) -> Ontology:
    """Load an overrides JSON and apply edge-label aliases on top of *ontology*.

    *overrides_path* must be absolute — relative-path resolution is the
    caller's responsibility (it needs the corpus TOML directory, which this
    module shouldn't care about).

    Returns a new ``Ontology``.  ``OntologyError`` from
    :meth:`~Ontology.with_edge_aliases` surfaces as-is; any other
    malformation raises :exc:`OntologyOverridesError`.
    """
    if not overrides_path.is_absolute():
        raise OntologyOverridesError(f"overrides path must be absolute; got {overrides_path}")
    try:
        raw_text = overrides_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OntologyOverridesError(
            f"overrides file not readable: {overrides_path} ({exc})"
        ) from exc
    except UnicodeDecodeError as exc:
        raise OntologyOverridesError(
            f"overrides file is not valid UTF-8: {overrides_path} ({exc})"
        ) from exc
    try:
        data: Any = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise OntologyOverridesError(
            f"overrides file is not valid JSON: {overrides_path} ({exc})"
        ) from exc
    if not isinstance(data, dict):
        raise OntologyOverridesError(
            f"overrides file top level must be a JSON object; "
            f"got {type(data).__name__} in {overrides_path}"
        )
    raw_edge_aliases = data.get("edge_label_aliases")
    if raw_edge_aliases is None:
        return ontology
    if not isinstance(raw_edge_aliases, dict):
        raise OntologyOverridesError(
            f"'edge_label_aliases' must be an object; "
            f"got {type(raw_edge_aliases).__name__} in {overrides_path}"
        )
    for k, v in raw_edge_aliases.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise OntologyOverridesError(
                f"'edge_label_aliases' keys and values must be strings; "
                f"got {k!r} → {v!r} in {overrides_path}"
            )
    if not raw_edge_aliases:
        return ontology
    # OntologyError from with_edge_aliases surfaces as-is — don't catch-and-wrap.
    return ontology.with_edge_aliases(raw_edge_aliases)
