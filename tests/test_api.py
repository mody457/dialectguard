"""End-to-end tests over the HTTP layer.

These go through the real app: routing, schema validation, preprocessing and
the loaded model. They assert on status codes and response shape rather than
on which dialect comes back, because at 61.8 percent macro F1 the predicted
label for any single example is not a stable contract.
"""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.config import (
    API_PREFIX,
    ID_TO_DIALECT,
    MAX_INPUT_CHARS,
    MAX_REQUEST_BODY_BYTES,
)
from app.main import app
from app.model import DialectClassifier

PREDICT_URL = f"{API_PREFIX}/predict"

# Short Gulf dialect samples. Ground-truth country is intentionally not
# asserted, only that each one produces a well-formed prediction.
GULF_SAMPLES = [
    "شلونك اليوم شخبارك عساك طيب",
    "وش رايك في هالموضوع يا خوي",
    "شحالك اليوم شو تسوي",
]


def assert_error_shape(body: dict, expected_code: str, expected_status: int) -> None:
    """Assert the canonical error envelope, with no leaked internals."""
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "status_code"}
    assert body["error"]["code"] == expected_code
    assert body["error"]["status_code"] == expected_status
    assert isinstance(body["error"]["message"], str)
    assert body["error"]["message"]
    assert "Traceback" not in body["error"]["message"]


def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_dir"]


def test_version_reports_metadata(client: TestClient) -> None:
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == "v1"
    assert body["base_model"] == "UBC-NLP/MARBERTv2"
    assert body["macro_f1"] == 0.618
    assert "git_commit" in body


@pytest.mark.parametrize("text", GULF_SAMPLES)
def test_predict_returns_dialect_and_confidence(client: TestClient, text: str) -> None:
    response = client.post(PREDICT_URL, json={"text": text})
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"dialect", "confidence"}
    assert body["dialect"] in ID_TO_DIALECT
    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_applies_preprocessing(client: TestClient) -> None:
    """Mentions, hashtags and URLs must not change the prediction.

    They are stripped before inference, so a decorated sample and its clean
    equivalent have to produce the identical result. A drift here means the
    serving pipeline no longer matches training.
    """
    clean = "شلونك اليوم شخبارك عساك طيب"
    decorated = f"@someuser {clean} #الكويت https://t.co/abc123"

    clean_body = client.post(PREDICT_URL, json={"text": clean}).json()
    decorated_body = client.post(PREDICT_URL, json={"text": decorated}).json()

    assert clean_body["dialect"] == decorated_body["dialect"]
    assert clean_body["confidence"] == pytest.approx(decorated_body["confidence"])


def test_predict_rejects_empty_text(client: TestClient) -> None:
    response = client.post(PREDICT_URL, json={"text": "   "})
    assert response.status_code == 422
    assert_error_shape(response.json(), "EMPTY_TEXT", 422)


def test_predict_rejects_non_arabic_text(client: TestClient) -> None:
    response = client.post(PREDICT_URL, json={"text": "hello how are you today"})
    assert response.status_code == 422
    assert_error_shape(response.json(), "NOT_ARABIC_DOMINANT", 422)


def test_predict_rejects_text_over_character_cap(client: TestClient) -> None:
    response = client.post(PREDICT_URL, json={"text": "ا" * (MAX_INPUT_CHARS + 1)})
    assert response.status_code == 422
    assert_error_shape(response.json(), "TEXT_TOO_LONG", 422)


def test_predict_rejects_text_over_token_budget(client: TestClient) -> None:
    # Under the character cap but well over the model's 128 token budget.
    response = client.post(PREDICT_URL, json={"text": "شلون " * 160})
    assert response.status_code == 422
    assert_error_shape(response.json(), "TEXT_TOO_MANY_TOKENS", 422)


def test_predict_rejects_text_of_only_artifacts(client: TestClient) -> None:
    response = client.post(PREDICT_URL, json={"text": "@user #tag https://example.com"})
    assert response.status_code == 422
    assert_error_shape(response.json(), "TEXT_EMPTY_AFTER_PREPROCESSING", 422)


def test_predict_rejects_missing_field(client: TestClient) -> None:
    response = client.post(PREDICT_URL, json={})
    assert response.status_code == 422
    assert_error_shape(response.json(), "INVALID_REQUEST_BODY", 422)


def test_unknown_route_returns_canonical_error(client: TestClient) -> None:
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    assert_error_shape(response.json(), "NOT_FOUND", 404)


def test_rejects_request_body_over_the_byte_cap(client: TestClient) -> None:
    """Oversized bodies are refused from the header, before being buffered."""
    oversized = "ا" * (MAX_REQUEST_BODY_BYTES // 2)
    response = client.post(PREDICT_URL, json={"text": oversized})
    assert response.status_code == 413
    assert_error_shape(response.json(), "REQUEST_BODY_TOO_LARGE", 413)


class _RecordCollector(logging.Handler):
    """Collects LogRecords so a test can assert on their structured context."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_prediction_log_omits_raw_text_by_default(client: TestClient) -> None:
    """The prediction log carries the required fields but not the input text.

    LOG_RAW_TEXT defaults to off because request bodies are user content. That
    default had no coverage, so flipping it would not have failed anything.
    """
    text = "شلونك اليوم شخبارك عساك طيب"
    collector = _RecordCollector()
    app_logger = logging.getLogger("app.main")
    app_logger.addHandler(collector)
    try:
        response = client.post(PREDICT_URL, json={"text": text})
    finally:
        app_logger.removeHandler(collector)

    assert response.status_code == 200

    contexts = [
        record.context
        for record in collector.records
        if getattr(record, "context", {}).get("event") == "prediction"
    ]
    assert len(contexts) == 1
    context = contexts[0]

    assert set(context) == {
        "event",
        "input_length",
        "dialect",
        "confidence",
        "latency_ms",
    }
    assert context["input_length"] == len(text)
    assert text not in json.dumps(context, ensure_ascii=False)


def test_unhandled_exception_returns_canonical_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected failure surfaces as INTERNAL_ERROR with no internals.

    This was the one error path the suite never exercised, despite being the
    one that leaks a traceback to callers if it ever regresses. The client
    fixture is required so the model is already loaded on the shared app.
    """

    def raise_unexpected(self: DialectClassifier, _cleaned_text: str) -> None:
        raise RuntimeError("synthetic failure carrying internal detail")

    monkeypatch.setattr(DialectClassifier, "predict", raise_unexpected)

    # The shared client re-raises server exceptions, which is the right default
    # for surfacing real bugs. This one returns the response a caller would
    # actually receive instead.
    non_raising = TestClient(app, raise_server_exceptions=False)
    response = non_raising.post(PREDICT_URL, json={"text": "شلونك اليوم"})

    assert response.status_code == 500
    assert_error_shape(response.json(), "INTERNAL_ERROR", 500)
    assert "synthetic failure" not in response.text
    assert "RuntimeError" not in response.text
