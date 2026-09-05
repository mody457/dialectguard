"""Tests for the per-client rate limit.

Two halves. The identity function is tested directly, because getting it wrong
is the failure that matters most: treating every caller as one bucket throttles
everyone, and trusting a spoofable header throttles no one. The rest goes
through the real HTTP path, since the limit is enforced in middleware and a
direct call would not exercise the ordering or the error envelope.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from app import rate_limit
from app.config import API_PREFIX

PREDICT_PATH = f"{API_PREFIX}/predict"

# Short and unambiguously Gulf, so a served request is a prediction rather than
# a validation rejection wearing the wrong status code.
ARABIC_TEXT = "شلونك اليوم شخبارك عساك طيب"


def build_request(
    client_host: str | None, headers: dict[str, str] | None = None
) -> Request:
    """Build a bare Request with a chosen peer address and headers."""
    raw_headers = [
        (key.lower().encode(), value.encode())
        for key, value in (headers or {}).items()
    ]
    scope: dict[str, object] = {
        "type": "http",
        "method": "POST",
        "path": PREDICT_PATH,
        "headers": raw_headers,
        "query_string": b"",
        "client": (client_host, 1234) if client_host else None,
    }
    return Request(scope)


@pytest.fixture
def trust_proxy() -> Iterator[None]:
    """Turn on X-Forwarded-For trust for one test."""
    original = rate_limit.TRUST_PROXY_HEADER
    rate_limit.TRUST_PROXY_HEADER = True
    yield
    rate_limit.TRUST_PROXY_HEADER = original


def test_identifier_uses_the_socket_peer() -> None:
    assert rate_limit.client_identifier(build_request("203.0.113.7")) == "203.0.113.7"


def test_identifier_ignores_forwarded_header_by_default() -> None:
    """The header is caller supplied, so honouring it would void the limit."""
    request = build_request("203.0.113.7", {"X-Forwarded-For": "198.51.100.1"})
    assert rate_limit.client_identifier(request) == "203.0.113.7"


def test_identifier_reads_forwarded_header_when_trusted(trust_proxy: None) -> None:
    request = build_request("10.0.0.1", {"X-Forwarded-For": "198.51.100.1"})
    assert rate_limit.client_identifier(request) == "198.51.100.1"


def test_identifier_takes_the_original_client_from_a_proxy_chain(
    trust_proxy: None,
) -> None:
    """Everything after the first entry is the proxy chain, not the caller."""
    request = build_request(
        "10.0.0.1", {"X-Forwarded-For": "198.51.100.1, 10.0.0.9, 10.0.0.8"}
    )
    assert rate_limit.client_identifier(request) == "198.51.100.1"


def test_identifier_falls_back_when_the_header_is_trusted_but_empty(
    trust_proxy: None,
) -> None:
    request = build_request("10.0.0.1", {"X-Forwarded-For": "   "})
    assert rate_limit.client_identifier(request) == "10.0.0.1"


def test_identifier_buckets_peerless_requests_together() -> None:
    """Throttling an unidentifiable caller beats exempting one."""
    assert rate_limit.client_identifier(build_request(None)) == (
        rate_limit.UNKNOWN_CLIENT
    )


def test_current_rate_limit_is_read_at_call_time() -> None:
    """The limit is a callable so it is not frozen into import-time state."""
    original = rate_limit.RATE_LIMIT
    rate_limit.RATE_LIMIT = "7/minute"
    try:
        assert rate_limit.current_rate_limit() == "7/minute"
    finally:
        rate_limit.RATE_LIMIT = original


@pytest.mark.parametrize("rate_limited", ["2/minute"], indirect=True)
def test_requests_under_the_limit_are_served(
    client: TestClient, rate_limited: None
) -> None:
    for _ in range(2):
        response = client.post(PREDICT_PATH, json={"text": ARABIC_TEXT})
        assert response.status_code == 200


@pytest.mark.parametrize("rate_limited", ["2/minute"], indirect=True)
def test_the_request_over_the_limit_is_refused(
    client: TestClient, rate_limited: None
) -> None:
    for _ in range(2):
        client.post(PREDICT_PATH, json={"text": ARABIC_TEXT})

    response = client.post(PREDICT_PATH, json={"text": ARABIC_TEXT})
    assert response.status_code == 429


@pytest.mark.parametrize("rate_limited", ["1/minute"], indirect=True)
def test_the_refusal_uses_the_canonical_error_envelope(
    client: TestClient, rate_limited: None
) -> None:
    """A throttled client must not meet a second error shape."""
    client.post(PREDICT_PATH, json={"text": ARABIC_TEXT})
    response = client.post(PREDICT_PATH, json={"text": ARABIC_TEXT})

    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "status_code"}
    assert body["error"]["code"] == rate_limit.RATE_LIMIT_ERROR_CODE
    assert body["error"]["status_code"] == 429


@pytest.mark.parametrize("rate_limited", ["1/minute"], indirect=True)
def test_the_refusal_carries_a_usable_retry_after(
    client: TestClient, rate_limited: None
) -> None:
    """Clients should be able to back off without parsing the message."""
    client.post(PREDICT_PATH, json={"text": ARABIC_TEXT})
    response = client.post(PREDICT_PATH, json={"text": ARABIC_TEXT})

    retry_after = int(response.headers["retry-after"])
    assert rate_limit.MIN_RETRY_AFTER_SECONDS <= retry_after <= 60


@pytest.mark.parametrize("rate_limited", ["1/minute"], indirect=True)
def test_health_and_version_stay_exempt(
    client: TestClient, rate_limited: None
) -> None:
    """The probes have to answer while the classification routes are closed."""
    client.post(PREDICT_PATH, json={"text": ARABIC_TEXT})
    assert client.post(PREDICT_PATH, json={"text": ARABIC_TEXT}).status_code == 429

    for _ in range(5):
        assert client.get("/health").status_code == 200
        assert client.get("/version").status_code == 200


@pytest.mark.parametrize("rate_limited", ["1/minute"], indirect=True)
def test_the_limit_is_per_client_not_global(rate_limited: None) -> None:
    """One noisy caller must not throttle everyone else.

    Built on a stub app rather than the real one. TestClient sends every
    request from the same address, so two identities need two apps or a key
    function that can tell them apart, and the second is the cheaper stub.
    """
    from slowapi.middleware import SlowAPIMiddleware

    stub = FastAPI()
    stub.state.limiter = rate_limit.limiter
    stub.add_exception_handler(
        rate_limit.RateLimitExceeded, rate_limit.handle_rate_limit_exceeded
    )
    stub.add_middleware(SlowAPIMiddleware)

    @stub.get("/echo")
    async def echo() -> dict[str, bool]:
        return {"ok": True}

    with TestClient(stub) as stub_client:
        first = {"X-Forwarded-For": "198.51.100.1"}
        second = {"X-Forwarded-For": "198.51.100.2"}

        original = rate_limit.TRUST_PROXY_HEADER
        rate_limit.TRUST_PROXY_HEADER = True
        try:
            assert stub_client.get("/echo", headers=first).status_code == 200
            assert stub_client.get("/echo", headers=first).status_code == 429
            # A different caller still has its own allowance.
            assert stub_client.get("/echo", headers=second).status_code == 200
        finally:
            rate_limit.TRUST_PROXY_HEADER = original


def test_the_configured_default_is_thirty_a_minute() -> None:
    """Pins the shipped ceiling, which the README documents."""
    from app import config

    assert config.RATE_LIMIT == "30/minute"
