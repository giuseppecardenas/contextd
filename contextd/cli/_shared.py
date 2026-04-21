"""Private helpers shared across the CLI sub-modules.

Houses the rich ``Console`` instance, the ``_load_cfg`` helper, and the
``PipelineDeps`` dataclass. Kept in a single private module so all
sub-modules import from one place — and the console singleton is shared,
not re-constructed per sub-module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console

from contextd._paths import contextd_home

if TYPE_CHECKING:
    from contextd.config import Config
    from contextd.indexer.hasher import FileHasher
    from contextd.inference.relate import RelationshipInferrer
    from contextd.inference.summarise import Summariser
    from contextd.providers.base import EmbeddingProvider
    from contextd.storage.base import GraphStore

console = Console()


@dataclass
class PipelineDeps:
    """Collaborators `index --bootstrap` assembles per invocation.

    Extracted so the wiring logic lives in a single function
    (_build_pipeline_deps) rather than inlined in the `index` command body.
    Keeps the `index` body focused on orchestration.
    """

    summariser: Summariser
    inferrer: RelationshipInferrer
    hasher: FileHasher
    embedder: EmbeddingProvider
    store: GraphStore


def _load_cfg() -> Config:
    """Load user config.toml with fallback to packaged default."""
    from contextd.config import Config

    path = contextd_home() / "config.toml"
    return Config.load(path) if path.exists() else Config.load_default()
