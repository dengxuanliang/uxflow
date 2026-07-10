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
