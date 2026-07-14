# SPDX-License-Identifier: Apache-2.0
from module1.pipeline import TrajectoryPipeline, PipelineConfig
from module1.index import MemoryIndex


class _StubGateway:
    async def call(self, messages, model):
        return None, {}


def test_default_store_is_memory_index():
    p = TrajectoryPipeline(config=PipelineConfig(), gateway=_StubGateway())
    assert isinstance(p._store, MemoryIndex)


def test_injected_store_factory_is_used():
    made = []

    def factory():
        s = MemoryIndex()
        made.append(s)
        return s

    p = TrajectoryPipeline(config=PipelineConfig(), gateway=_StubGateway(),
                           store_factory=factory)
    assert p._store is made[-1]
    # per-run reset must also use the factory, not a hardcoded MemoryIndex
    p._reset_store()
    assert p._store is made[-1] and len(made) == 2
