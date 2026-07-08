import json
import pathlib
import pytest

FIXTURES_DIR = pathlib.Path(__file__).parent.parent.parent / "fixtures"


@pytest.fixture
def trajectories_path():
    return FIXTURES_DIR / "trajectories" / "sample_01.jsonl"


@pytest.fixture
def sample_trajectories(trajectories_path):
    trajectories = []
    with open(trajectories_path) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    return trajectories
