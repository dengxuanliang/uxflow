"""UXFlow embedding backends behind a single Embedder protocol."""

from uxflow_embed.protocol import Embedder
from uxflow_embed.fake import FakeEmbedder

__all__ = ["Embedder", "FakeEmbedder"]
