import pytest

from llm_gateway.config import GatewayConfig


@pytest.fixture
def default_config():
    """Test-friendly config: low timeouts, small windows, fast cooldown."""
    return GatewayConfig(
        litellm_base="http://localhost:9999/v1",
        litellm_key="test-key",
        request_timeout=5,
        request_timeout_escalation=[3, 5, 8],
        concurrency=10,
        rps_limit=100.0,
        adaptive_window_requests=5,
        adaptive_window_seconds=2.0,
        adaptive_open_base_cooldown=1.0,
        adaptive_open_max_cooldown=3.0,
        adaptive_min_observations_degraded=2,
        adaptive_min_observations_open=4,
        adaptive_min_failures_degraded=1,
        adaptive_min_failures_open=2,
        fatal_abort_global_rate_min_obs=10,
    )
