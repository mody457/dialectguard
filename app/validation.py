"""Input validation applied before any text reaches the tokenizer.

The model has six output classes and no "not applicable" option, so anything
that is not short Gulf Arabic text has to be rejected here. Without these
gates the API would answer an English sentence with a confident country code.

Every rejection is a 422 carrying a distinct machine-readable code, so callers
can tell a too-long input apart from a non-Arabic one without string matching.
"""

from __future__ import annotations

from fastapi import status

from app.config import MAX_INPUT_CHARS, MIN_ARABIC_CHAR_RATIO
from app.errors import APIError
from app.preprocessing import arabic_ratio, preprocess

_UNPROCESSABLE = status.HTTP_422_UNPROCESSABLE_CONTENT


def validate_and_clean(raw_text: str) -> str:
    """Validate raw request text and return its preprocessed form.

    The length cap is checked against the raw string, before preprocessing, so
    that a megabyte of hashtags cannot be spent on regex work first.

    Raises:
        APIError: if the text is empty, too long, empty after cleaning, or not
            predominantly Arabic script.
    """
    if not raw_text.strip():
        raise APIError(
            _UNPROCESSABLE,
            "EMPTY_TEXT",
            "Field 'text' must contain at least one non-whitespace character.",
        )

    if len(raw_text) > MAX_INPUT_CHARS:
        raise APIError(
            _UNPROCESSABLE,
            "TEXT_TOO_LONG",
            f"Field 'text' is {len(raw_text)} characters, "
            f"the maximum is {MAX_INPUT_CHARS}.",
        )

    cleaned = preprocess(raw_text)
    if not cleaned:
        raise APIError(
            _UNPROCESSABLE,
            "TEXT_EMPTY_AFTER_PREPROCESSING",
            "Field 'text' contained only mentions, hashtags or URLs, "
            "which are removed before inference.",
        )

    ratio = arabic_ratio(cleaned)
    if ratio < MIN_ARABIC_CHAR_RATIO:
        raise APIError(
            _UNPROCESSABLE,
            "NOT_ARABIC_DOMINANT",
            f"Field 'text' is {ratio:.0%} Arabic script after cleaning, "
            f"the minimum is {MIN_ARABIC_CHAR_RATIO:.0%}. "
            "This model only classifies Arabic text.",
        )

    return cleaned
