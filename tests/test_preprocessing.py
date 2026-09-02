"""Tests for the training-time cleaning pipeline.

These guard against train/serve skew. If a step here is weakened, real-world
accuracy drops below the documented 61.8 percent macro F1 without any test
failing elsewhere.
"""

from __future__ import annotations

import pytest

from app.preprocessing import arabic_ratio, preprocess

ARABIC_SENTENCE = "شلونك اليوم"


def test_strips_mentions() -> None:
    assert preprocess(f"@user_123 {ARABIC_SENTENCE}") == ARABIC_SENTENCE


def test_strips_hashtags() -> None:
    assert preprocess(f"{ARABIC_SENTENCE} #الكويت_اليوم") == ARABIC_SENTENCE


def test_strips_urls() -> None:
    assert (
        preprocess(f"{ARABIC_SENTENCE} https://example.com/a/b?c=1") == ARABIC_SENTENCE
    )
    assert preprocess(f"www.example.com {ARABIC_SENTENCE}") == ARABIC_SENTENCE


def test_normalizes_whitespace() -> None:
    assert preprocess("  شلونك\t\tاليوم\n\n ") == ARABIC_SENTENCE


def test_strips_all_artifacts_together() -> None:
    raw = f"@ahmad  {ARABIC_SENTENCE}   #دبي http://t.co/xyz  @sara"
    assert preprocess(raw) == ARABIC_SENTENCE


def test_leaves_clean_text_untouched() -> None:
    assert preprocess(ARABIC_SENTENCE) == ARABIC_SENTENCE


def test_preserves_arabic_punctuation_outside_hashtags() -> None:
    assert preprocess("شلونك، زين؟") == "شلونك، زين؟"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("شلونك اليوم", 1.0),
        ("hello world", 0.0),
        ("", 0.0),
        ("123 !!! 😀", 0.0),
        ("شلون ok", 4 / 6),
    ],
)
def test_arabic_ratio(text: str, expected: float) -> None:
    assert arabic_ratio(text) == pytest.approx(expected)


def test_arabic_ratio_ignores_digits_and_emoji() -> None:
    # Digits and emoji must not dilute the ratio, or ordinary tweets would be
    # rejected as non-Arabic.
    assert arabic_ratio("شلونك 2024 😀😀😀") == pytest.approx(1.0)
