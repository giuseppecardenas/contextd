"""Transport-independent search orchestration (hybrid ranking).

Kept out of ``contextd/mcp/`` so a future CLI or HTTP caller can reuse the
fusion logic without importing the MCP package, and so the CI
abstraction-invariant grep (which excludes only ``contextd/storage/``)
forbids any backend import from this package.
"""

from __future__ import annotations

from contextd.search.fusion import flatten_row, reciprocal_rank_fusion

__all__ = ["flatten_row", "reciprocal_rank_fusion"]
