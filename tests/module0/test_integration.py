"""Integration tests: Module 0 → LLMGateway → LiteLLM.

Requires a running LiteLLM proxy. Skipped unless MODULE0_INTEGRATION=1.

Setup:
    1. Fill in .env at project root:
         MODULE0_INTEGRATION=1
         LITELLM_BASE=http://localhost:4000/v1
         LITELLM_KEY=sk-...
         MODULE0_TEST_MODEL=gpt-4o-mini
    2. pytest tests/module0/test_integration.py -v
"""

import os
import pathlib

import pytest

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent.parent / ".env")

from llm_gateway import LLMGateway, GatewayConfig  # noqa: E402  (import after load_dotenv)
from module0 import QueryCompiler, Taxonomy  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.environ.get("MODULE0_INTEGRATION"),
    reason="set MODULE0_INTEGRATION=1 and a running LiteLLM to run",
)

MODEL = os.environ.get("MODULE0_TEST_MODEL", "gpt-4o-mini")


@pytest.fixture
def taxonomy():
    p = pathlib.Path(__file__).parent.parent.parent / "fixtures" / "taxonomy_v0.json"
    return Taxonomy.load(p)


def _config():
    return GatewayConfig(
        litellm_base=os.environ.get("LITELLM_BASE", "http://localhost:4000/v1"),
        litellm_key=os.environ.get("LITELLM_KEY", ""),
        concurrency=5,
        transport_stuck_seconds=0,
    )


async def test_end_to_end_simple(taxonomy):
    """Real LLM: single clear problem → valid Problem Spec."""
    async with LLMGateway(_config()) as gw:
        compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model=MODEL, embedding_model=None)
        spec = await compiler.compile("写入py文件经常有语法错误")

        assert spec.domain == "agentic_swe"
        assert len(spec.sub_problems) >= 1
        for sp in spec.sub_problems:
            assert 1 <= len(sp.target_capability) <= 3
            assert 2 <= len(sp.hyde_positive) <= 3
            assert sp.confidence >= 0.8
            assert sp.route == "pass"


async def test_end_to_end_ambiguous(taxonomy):
    """Real LLM: the p12 '中途停止' benchmark case."""
    async with LLMGateway(_config()) as gw:
        compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model=MODEL, embedding_model=None)
        spec = await compiler.compile("解题过程中途停止")
        for sp in spec.sub_problems:
            if sp.origin == "clarified":
                assert sp.parent_id is not None


@pytest.mark.requires_model
async def test_embedding_integration(taxonomy):
    """Real LLM + real embedding model end-to-end."""
    from module0.embedding import EmbeddingModel
    emb = EmbeddingModel()
    async with LLMGateway(_config()) as gw:
        compiler = QueryCompiler(gateway=gw, taxonomy=taxonomy, model=MODEL, embedding_model=emb)
        spec = await compiler.compile("写入py文件经常有语法错误")
        assert spec is not None
