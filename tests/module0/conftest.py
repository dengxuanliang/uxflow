import json
import pathlib
import pytest

# Load .env before any test module imports sentence-transformers / huggingface,
# so HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are already in os.environ when the
# model loads. conftest runs before collection, so this covers both
# test_embedding.py (no dotenv of its own) and test_integration.py.
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent.parent / ".env")

FIXTURES_DIR = pathlib.Path(__file__).parent.parent.parent / "fixtures"


@pytest.fixture
def taxonomy_v0_path():
    return FIXTURES_DIR / "taxonomy_v0.json"


@pytest.fixture
def taxonomy_v0(taxonomy_v0_path):
    with open(taxonomy_v0_path) as f:
        return json.load(f)


@pytest.fixture
def golden_problem_spec():
    path = FIXTURES_DIR / "problem_specs" / "test_01.json"
    with open(path) as f:
        return json.load(f)
