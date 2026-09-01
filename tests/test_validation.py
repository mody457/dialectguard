"""Tests for the pre-inference input gates."""

from __future__ import annotations

import pytest

from app.config import MAX_INPUT_CHARS
from app.errors import APIError
from app.validation import validate_and_clean


def test_returns_cleaned_text_for_valid_input() -> None:
    assert validate_and_clean("@ali شلونك اليوم") == "شلونك اليوم"


@pytest.mark.parametrize("raw", ["", "   ", "\n\t"])
def test_rejects_empty_text(raw: str) -> None:
    with pytest.raises(APIError) as exc_info:
        validate_and_clean(raw)
    assert exc_info.value.code == "EMPTY_TEXT"
    assert exc_info.value.status_code == 422


def test_rejects_text_over_character_cap() -> None:
    with pytest.raises(APIError) as exc_info:
        validate_and_clean("ا" * (MAX_INPUT_CHARS + 1))
    assert exc_info.value.code == "TEXT_TOO_LONG"


def test_rejects_text_that_is_only_artifacts() -> None:
    with pytest.raises(APIError) as exc_info:
        validate_and_clean("@user #tag https://example.com")
    assert exc_info.value.code == "TEXT_EMPTY_AFTER_PREPROCESSING"


def test_rejects_non_arabic_text() -> None:
    with pytest.raises(APIError) as exc_info:
        validate_and_clean("how are you doing today my friend")
    assert exc_info.value.code == "NOT_ARABIC_DOMINANT"


def test_accepts_mixed_text_that_is_mostly_arabic() -> None:
    assert validate_and_clean("شلونك اليوم ok") == "شلونك اليوم ok"
