"""UXFlow embedding backends behind a single Embedder protocol."""

from uxflow_embed.protocol import Embedder
from uxflow_embed.fake import FakeEmbedder
from uxflow_embed.local import LocalEmbedder
from uxflow_embed.api import ApiEmbedder

__all__ = ["Embedder", "FakeEmbedder", "LocalEmbedder", "ApiEmbedder"]
