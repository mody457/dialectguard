"""FastAPI application for Gulf Arabic dialect classification.

Wires together the request schemas, the training-time preprocessing pipeline
and the fine-tuned MARBERTv2 classifier. The model is loaded once during
startup and shared across requests through application state.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request, status
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.config import (
    API_PREFIX,
    API_VERSION,
    GIT_COMMIT,
    ID_TO_DIALECT,
    LOG_RAW_TEXT,
    MAX_INPUT_CHARS,
    MAX_SEQUENCE_LENGTH,
    MODEL_DIR,
    MODEL_MACRO_F1,
    MODEL_VERSION,
)
from app.errors import APIError, register_exception_handlers
from app.logging_config import configure_logging
from app.model import DialectClassifier
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    VersionResponse,
)
from app.validation import validate_and_clean

logger = logging.getLogger(__name__)

BASE_MODEL = "UBC-NLP/MARBERTv2"

API_DESCRIPTION = f"""
Classifies Arabic text into one of six Gulf dialects: {", ".join(ID_TO_DIALECT)}.

Model: MARBERTv2 fine-tuned on the Gulf subset of QADI. Macro F1 on the
held-out test set is 61.8 percent, against 44.6 percent for a TF-IDF and
logistic regression baseline.

Known limitations, which callers should design around:

- Training data is Twitter/X only. Behaviour on WhatsApp messages, product
  reviews and forum posts is unverified.
- Residual user-level leakage in the train/test split cannot be ruled out.
  The source dataset carries no user_id, so only exact-duplicate text leakage
  was eliminated.
- The training data is imbalanced, with Kuwait overrepresented relative to
  Oman. Per-country performance varies.
- At 61.8 percent macro F1, roughly one prediction in three is wrong. Do not
  use this for high-stakes routing without a confidence threshold or a human
  review step.

Input is cleaned before inference: mentions, hashtags and URLs are stripped
and whitespace is normalized, matching the training pipeline exactly.
""".strip()

ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    422: {"model": ErrorResponse, "description": "Input rejected before inference."},
    503: {"model": ErrorResponse, "description": "Model is not loaded."},
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model before the first request and release it on shutdown.

    A failure here is deliberately fatal. A dialect API that boots without its
    weights would answer health checks while being useless, so it is better to
    crash and let the supervisor restart or roll back.
    """
    configure_logging()
    app.state.classifier = DialectClassifier.load(MODEL_DIR)
    try:
        yield
    finally:
        app.state.classifier = None


app = FastAPI(
    title="DialectGuard",
    description=API_DESCRIPTION,
    version=__version__,
    lifespan=lifespan,
)
register_exception_handlers(app)

router = APIRouter(prefix=API_PREFIX, tags=["prediction"])


def get_classifier(request: Request) -> DialectClassifier:
    """Return the loaded classifier, or fail with a 503 if it is missing."""
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None:
        raise APIError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "MODEL_NOT_LOADED",
            "The classification model is not loaded. The service is not ready.",
        )
    return classifier


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses=ERROR_RESPONSES,
    summary="Classify Arabic text into a Gulf dialect",
)
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Classify a single piece of Arabic text.

    Inference runs in a worker thread because the forward pass is blocking and
    would otherwise stall the event loop for every other in-flight request.
    """
    classifier = get_classifier(request)
    started = time.perf_counter()

    cleaned_text = validate_and_clean(payload.text)
    prediction = await run_in_threadpool(classifier.predict, cleaned_text)

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    context: dict[str, object] = {
        "event": "prediction",
        "input_length": len(payload.text),
        "dialect": prediction.dialect,
        "confidence": round(prediction.confidence, 4),
        "latency_ms": latency_ms,
    }
    if LOG_RAW_TEXT:
        context["text"] = payload.text
    logger.info("prediction", extra={"context": context})

    return PredictResponse(dialect=prediction.dialect, confidence=prediction.confidence)


app.include_router(router)


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: ERROR_RESPONSES[503]},
    tags=["operations"],
    summary="Readiness check",
)
async def health(request: Request) -> HealthResponse:
    """Report whether the model is loaded, not merely whether the process runs.

    Use this as the readiness probe. A process that is up but has no weights
    in memory cannot serve traffic and should not receive any.
    """
    classifier = get_classifier(request)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_dir=str(classifier.model_dir),
    )


@app.get(
    "/version",
    response_model=VersionResponse,
    tags=["operations"],
    summary="Build and model metadata",
)
async def version() -> VersionResponse:
    """Return the versions a caller needs to reproduce or debug a prediction."""
    return VersionResponse(
        api_version=API_VERSION,
        model_version=MODEL_VERSION,
        base_model=BASE_MODEL,
        git_commit=GIT_COMMIT,
        macro_f1=MODEL_MACRO_F1,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        max_input_chars=MAX_INPUT_CHARS,
    )
