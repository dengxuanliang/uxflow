from uxflow_embed import Embedder


class _Impl:
    @property
    def dimension(self) -> int:
        return 3

    def embed(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


def test_embedder_is_runtime_checkable_protocol():
    assert isinstance(_Impl(), Embedder)


def test_non_impl_fails_isinstance():
    class Missing:
        def embed(self, text): ...
    assert not isinstance(Missing(), Embedder)


def test_preferred_batch_size_is_optional():
    """Implementations without preferred_batch_size still satisfy the Protocol."""
    from uxflow_embed.protocol import Embedder

    class MinimalEmbedder:
        @property
        def dimension(self) -> int:
            return 8

        def embed(self, text: str) -> list[float]:
            return [0.0] * 8

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 8] * len(texts)

    # Should NOT raise — preferred_batch_size is optional
    assert isinstance(MinimalEmbedder(), Embedder)

    # getattr with default should work
    emb = MinimalEmbedder()
    assert getattr(emb, "preferred_batch_size", 99) == 99
