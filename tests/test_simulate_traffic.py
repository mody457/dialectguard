"""Tests for the traffic simulator.

No server and no model. The simulator's job is to record what came back,
including refusals, and the failure worth guarding against is a refused request
being dropped or recorded as a prediction. A run that quietly lost its 422s
would show an English category with no rows and read as clean.
"""

from __future__ import annotations

import httpx
import pytest

from monitoring import samples, simulate_traffic

API_URL = "http://testserver"


def client_returning(handler) -> httpx.Client:
    """A client whose every request is answered by handler."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def ok_prediction(request: httpx.Request) -> httpx.Response:
    if request.url.path == simulate_traffic.HEALTH_PATH:
        return httpx.Response(200, json={"status": "ok", "model_loaded": True})
    return httpx.Response(200, json={"dialect": "KW", "confidence": 0.87})


def refused(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        422,
        json={
            "error": {
                "code": "NOT_ARABIC_DOMINANT",
                "message": "This model only classifies Arabic text.",
                "status_code": 422,
            }
        },
    )


def test_check_health_accepts_a_loaded_model() -> None:
    with client_returning(ok_prediction) as client:
        simulate_traffic.check_health(client, API_URL)


def test_check_health_rejects_a_process_with_no_model() -> None:
    """A reachable process is not readiness. 503s would look like drift."""

    def unloaded(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "loading", "model_loaded": False})

    with (
        client_returning(unloaded) as client,
        pytest.raises(RuntimeError, match="not loaded"),
    ):
        simulate_traffic.check_health(client, API_URL)


def test_check_health_says_how_to_start_the_service() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with (
        client_returning(refuse) as client,
        pytest.raises(RuntimeError, match="uvicorn"),
    ):
        simulate_traffic.check_health(client, API_URL)


def test_send_one_records_a_served_prediction() -> None:
    with client_returning(ok_prediction) as client:
        record = simulate_traffic.send_one(client, API_URL, "in_distribution", "hello")

    assert record.status_code == 200
    assert record.dialect == "KW"
    assert record.confidence == pytest.approx(0.87)
    assert record.error_code is None
    assert record.split == "reference"
    assert record.text_length == len("hello")


def test_send_one_records_a_refusal_without_inventing_a_prediction() -> None:
    """A 422 has to keep dialect and confidence empty, not default them."""
    with client_returning(refused) as client:
        record = simulate_traffic.send_one(client, API_URL, "english", "hello")

    assert record.status_code == 422
    assert record.error_code == "NOT_ARABIC_DOMINANT"
    assert record.dialect is None
    assert record.confidence is None
    assert record.split == "current"


def test_send_one_keeps_going_when_the_connection_drops() -> None:
    """A dropped connection is a row, so a partial run is still readable."""

    def drop(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with client_returning(drop) as client:
        record = simulate_traffic.send_one(client, API_URL, "msa", "hello")

    assert record.status_code == simulate_traffic.TRANSPORT_ERROR_STATUS
    assert record.error_code == simulate_traffic.TRANSPORT_ERROR_CODE
    assert record.dialect is None


def test_extract_error_code_falls_back_on_an_unexpected_body() -> None:
    """A stack trace page still has to be recorded, not raise in the recorder."""
    response = httpx.Response(500, text="<html>Internal Server Error</html>")
    assert simulate_traffic.extract_error_code(response) == "HTTP_500"


def test_run_category_sends_every_sample_and_tags_the_split() -> None:
    with client_returning(refused) as client:
        records = simulate_traffic.run_category(
            client, API_URL, "english", 25, samples.RANDOM_SEED
        )

    assert len(records) == 25
    assert {record.split for record in records} == {"current"}
    assert {record.category for record in records} == {"english"}


def test_records_to_frame_keeps_the_columns_the_report_reads() -> None:
    with client_returning(ok_prediction) as client:
        records = simulate_traffic.run_category(
            client, API_URL, "english", 5, samples.RANDOM_SEED
        )

    frame = simulate_traffic.records_to_frame(records)
    assert set(frame.columns) >= {
        "category",
        "split",
        "text",
        "text_length",
        "status_code",
        "dialect",
        "confidence",
        "error_code",
        "latency_ms",
    }
    assert len(frame) == 5


def test_summarize_counts_refusals_separately_from_served() -> None:
    with client_returning(refused) as client:
        records = simulate_traffic.run_category(
            client, API_URL, "english", 10, samples.RANDOM_SEED
        )

    summary = simulate_traffic.summarize(records)
    assert "10 sent, 0 served, 10 refused" in summary
    assert "NOT_ARABIC_DOMINANT: 10" in summary
