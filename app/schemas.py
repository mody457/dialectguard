"""Request and response models for the public API.

Endpoint signatures use these types rather than raw dicts so that the schema
is validated at the boundary and published in the generated OpenAPI document.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.config import ID_TO_DIALECT, MAX_INPUT_CHARS


class PredictRequest(BaseModel):
    """A single piece of Arabic text to classify."""

    text: str = Field(
        ...,
        description="Arabic text to classify. Cleaned before inference.",
        json_schema_extra={"example": "شلونك اليوم شخبارك عساك طيب"},
    )


class PredictResponse(BaseModel):
    """The predicted dialect and the model's confidence in it.

    Confidence is the softmax probability of the winning class. It is not a
    calibrated probability of correctness, but it is monotonic enough to use
    as a routing threshold.
    """

    dialect: str = Field(
        ...,
        description=f"Predicted Gulf dialect, one of: {', '.join(ID_TO_DIALECT)}.",
        json_schema_extra={"example": "KW"},
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Softmax probability of the predicted dialect.",
        json_schema_extra={"example": 0.87},
    )


class HealthResponse(BaseModel):
    """Readiness of the service, based on the model rather than the process."""

    # "model_" is a reserved Pydantic namespace. These field names are part
    # of the published API contract, so the guard is disabled instead.
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="'ok' when the model is ready to serve.")
    model_loaded: bool = Field(..., description="True if model weights are in memory.")
    model_dir: str = Field(..., description="Directory the model was loaded from.")


class VersionResponse(BaseModel):
    """Build and model metadata, for correlating predictions with a release."""

    model_config = ConfigDict(protected_namespaces=())

    api_version: str
    model_version: str
    base_model: str
    git_commit: str | None = Field(
        None, description="Short commit hash, null when not available."
    )
    macro_f1: float = Field(
        ..., description="Documented macro F1 on the held-out test set."
    )
    max_sequence_length: int
    max_input_chars: int = Field(
        MAX_INPUT_CHARS, description="Hard character cap on request text."
    )


class ErrorDetail(BaseModel):
    """The body of a failed request."""

    code: str = Field(..., description="Stable machine-readable error code.")
    message: str = Field(..., description="Human-readable explanation.")
    status_code: int = Field(..., description="HTTP status code, repeated for clients.")


class ErrorResponse(BaseModel):
    """Envelope returned by every failing endpoint."""

    error: ErrorDetail
