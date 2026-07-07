import asyncio

import pytest

from llm_gateway.outcomes import (
    OutcomeClass,
    RequestOutcome,
    classify_http_result,
    classify_exception,
)


def test_2xx_success():
    o = classify_http_result(200)
    assert o.classification is OutcomeClass.SUCCESS
    assert o.is_success
    assert not o.is_retryable


def test_2xx_parse_error_is_abnormal():
    o = classify_http_result(200, parse_error=True)
    assert o.classification is OutcomeClass.ABNORMAL_RESPONSE
    assert o.is_abnormal
    assert o.is_retryable


def test_429_is_overload():
    o = classify_http_result(429)
    assert o.classification is OutcomeClass.OVERLOAD
    assert o.is_overload
    assert o.is_infra_failure


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_5xx_is_server_error(code):
    o = classify_http_result(code)
    assert o.classification is OutcomeClass.SERVER_ERROR
    assert o.is_overload


@pytest.mark.parametrize("code", [401, 402])
def test_auth_errors(code):
    o = classify_http_result(code)
    assert o.classification is OutcomeClass.AUTH_ERROR
    assert not o.is_retryable


def test_400_content_filter_keyword():
    o = classify_http_result(400, error_text="Azure content_filter triggered")
    assert o.classification is OutcomeClass.CONTENT_FILTERED


def test_400_content_filter_explicit_flag():
    o = classify_http_result(400, content_filtered=True)
    assert o.classification is OutcomeClass.CONTENT_FILTERED


def test_400_context_length_is_input_error():
    o = classify_http_result(400, error_text="context_length_exceeded: too long")
    assert o.classification is OutcomeClass.INPUT_ERROR


def test_400_other_is_transient():
    o = classify_http_result(400, error_text="Unknown model foo")
    assert o.classification is OutcomeClass.TRANSIENT_ERROR
    assert o.is_retryable


def test_403_is_transient():
    o = classify_http_result(403)
    assert o.classification is OutcomeClass.TRANSIENT_ERROR


def test_other_4xx_is_input_error():
    o = classify_http_result(404)
    assert o.classification is OutcomeClass.INPUT_ERROR


def test_classify_timeout_exception():
    o = classify_exception(asyncio.TimeoutError())
    assert o.classification is OutcomeClass.TIMEOUT


def test_classify_httpx_timeout():
    import httpx
    o = classify_exception(httpx.ReadTimeout("slow"))
    assert o.classification is OutcomeClass.TIMEOUT


def test_classify_generic_exception():
    o = classify_exception(ValueError("boom"))
    assert o.classification is OutcomeClass.TRANSIENT_ERROR
    assert "ValueError" in o.error


def test_request_outcome_default_extra_is_isolated():
    a = RequestOutcome(classification=OutcomeClass.SUCCESS)
    b = RequestOutcome(classification=OutcomeClass.SUCCESS)
    a.extra["k"] = 1
    assert b.extra == {}
