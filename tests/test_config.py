import llm_gateway.config as cfg
from llm_gateway.config import GatewayConfig, resolve_httpx_connection_limits


def test_defaults():
    c = GatewayConfig()
    assert c.litellm_base == "http://localhost:4000/v1"
    assert c.concurrency == 200
    assert c.rps_limit == 20.0
    assert c.max_request_tokens == 10000
    assert c.enable_adaptive_runtime is True
    assert c.adaptive_open_base_cooldown == 15.0
    assert c.adaptive_probe_success_required == 3
    assert c.adaptive_probe_failure_tolerance == 3
    assert c.fatal_abort_global_rate_min_obs == 200


def test_timeout_escalation_default_factory_is_isolated():
    a = GatewayConfig()
    b = GatewayConfig()
    assert a.request_timeout_escalation == [60, 90, 120]
    a.request_timeout_escalation.append(999)
    assert b.request_timeout_escalation == [60, 90, 120]


def test_override_fields():
    c = GatewayConfig(concurrency=500, rps_limit=30.0)
    assert c.concurrency == 500
    assert c.rps_limit == 30.0


def test_resolve_limits_normal(monkeypatch):
    monkeypatch.setattr(cfg, "_soft_nofile_limit", lambda: 1_000_000)
    max_conn, keepalive, capped = resolve_httpx_connection_limits(
        requested_concurrency=200, extra_connections=10
    )
    assert max_conn == 210
    assert keepalive == 200
    assert capped is False


def test_resolve_limits_capped_by_fds(monkeypatch):
    monkeypatch.setattr(cfg, "_soft_nofile_limit", lambda: 128)
    max_conn, keepalive, capped = resolve_httpx_connection_limits(
        requested_concurrency=500, extra_connections=10, reserve_fds=64
    )
    assert max_conn == 64
    assert keepalive <= max_conn
    assert capped is True


def test_resolve_limits_none_soft_limit(monkeypatch):
    monkeypatch.setattr(cfg, "_soft_nofile_limit", lambda: None)
    max_conn, keepalive, capped = resolve_httpx_connection_limits(
        requested_concurrency=50
    )
    assert capped is False
    assert max_conn >= 50
