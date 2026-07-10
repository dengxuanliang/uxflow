"""Back-compat shim. Canonical location is uxflow_embed.LocalEmbedder.

Kept so existing imports (`from module0.embedding import EmbeddingModel`)
keep working after the Embedder extraction.
"""

from __future__ import annotations

from uxflow_embed import LocalEmbedder as EmbeddingModel

__all__ = ["EmbeddingModel"]
