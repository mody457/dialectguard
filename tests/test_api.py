"""End-to-end tests over the HTTP layer.

These go through the real app: routing, schema validation, preprocessing and
the loaded model. They assert on status codes and response shape rather than
on which dialect comes back, because at 61.8 percent macro F1 the predicted
label for any single example is not a stable contract.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import API_PREFIX, ID_TO_DIALECT, MAX_INPUT_CHARS

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
