import pytest


def test_local_embedder_importable_without_torch():
    # Import must not require torch at module load (lazy import in __init__).
    from uxflow_embed import LocalEmbedder
    assert LocalEmbedder is not None


def test_backcompat_alias_exists():
    # Existing code imports EmbeddingModel from module0.embedding.
    from module0.embedding import EmbeddingModel
    from uxflow_embed import LocalEmbedder
    assert EmbeddingModel is LocalEmbedder


def test_instantiation_without_torch_raises_clear_error():
    from uxflow_embed import LocalEmbedder
    try:
        import sentence_transformers  # noqa: F401
        pytest.skip("sentence-transformers installed; cannot test missing-dep path")
    except ImportError:
        pass
    with pytest.raises(ImportError) as exc:
        LocalEmbedder()
    assert "local-embed" in str(exc.value)


def test_local_files_only_param_defaults_false():
    import inspect
    from uxflow_embed import LocalEmbedder
    sig = inspect.signature(LocalEmbedder.__init__)
    assert sig.parameters["local_files_only"].default is False


def test_preferred_batch_size():
    """LocalEmbedder declares batch 32, matching its internal encode batch_size.

    Read off an uninitialized instance rather than a constructed one: the value
    is a constant that touches no instance state, while LocalEmbedder() would
    load a ~1.1GB model (and hit the network on a cold HF cache). Going through
    __get__ on a real instance -- rather than calling .fget() directly -- keeps
    this honest: if the property ever starts reading self, it raises here
    instead of silently passing. No skip guard needed; nothing imports torch.
    """
    from uxflow_embed import LocalEmbedder
    uninitialized = object.__new__(LocalEmbedder)
    assert uninitialized.preferred_batch_size == 32
