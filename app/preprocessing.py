"""Inference-time text preprocessing.

This module exists to prevent train/serve skew. The fine-tuning data was
cleaned by stripping mentions, hashtags and URLs before tokenization, after a
bug was found where the model was learning account identity from @mentions
instead of dialect. Serving has to apply the exact same transformation, in the
same order, or real-world accuracy drops below the documented 61.8% macro F1.

Do not relax or reorder these steps.
"""

from __future__ import annotations

import re

# Twitter-style handle. Unicode \w is intentional: some captured handles in
# the source data are not pure ASCII.
MENTION_PATTERN = re.compile(r"@\w+")

# Hashtag body, including the underscores common in Arabic hashtags
# (for example #الكويت_اليوم). Trailing Arabic punctuation is left in place.
HASHTAG_PATTERN = re.compile(r"#\w+")

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

WHITESPACE_PATTERN = re.compile(r"\s+")

# Contiguous ranges covering Arabic script, including the presentation forms
# that show up in text copied out of older systems.
ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)


def preprocess(text: str) -> str:
    """Apply the training-time cleaning pipeline to a single input string.

    The step order matches the documented training pipeline: mentions, then
    hashtags, then URLs, then whitespace normalization.
    """
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def is_arabic_char(char: str) -> bool:
    """Return True if the character belongs to an Arabic script block."""
    code_point = ord(char)
    return any(start <= code_point <= end for start, end in ARABIC_RANGES)


def arabic_ratio(text: str) -> float:
    """Return the share of alphabetic characters that are Arabic script.

    Digits, punctuation, emoji and whitespace are excluded from both sides of
    the ratio so that a normal tweet full of emoji is not misjudged as
    non-Arabic. Text with no alphabetic characters at all scores 0.0.
    """
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    arabic_letters = sum(1 for char in letters if is_arabic_char(char))
    return arabic_letters / len(letters)
