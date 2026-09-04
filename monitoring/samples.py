"""Sample sources for the drift simulation.

Four categories, each asking a different question about traffic the model was
never trained on. in_distribution is the reference every other category is
compared against, and it is the only one backed by a versioned file. The rest
exist to produce a single report, so they are fetched or generated on demand
and never committed or DVC tracked.

Loaders sit behind a registry rather than being called directly. That lets
simulate_traffic walk the categories without knowing where any of them come
from, and makes a source that is not wired up yet fail immediately, with a
message naming what it needs, instead of part way through a run.
"""

from __future__ import annotations

import random
from typing import Protocol

import pandas as pd

from app.config import PROJECT_ROOT

# Large enough for the drift statistics to mean something, small enough that a
# full simulation is a couple of minutes of local inference.
SAMPLES_PER_CATEGORY = 300

# Seeded so two runs of the same category compare like with like. A reference
# set that moved between runs would make every report incomparable to the last.
RANDOM_SEED = 42

# in_distribution is the reference: held-out Gulf text is what the documented
# 61.8% macro F1 was measured on. The other three are the shifted traffic.
REFERENCE_CATEGORY = "in_distribution"
CURRENT_CATEGORIES: tuple[str, ...] = ("msa", "non_gulf_dialect", "english")
ALL_CATEGORIES: tuple[str, ...] = (REFERENCE_CATEGORY, *CURRENT_CATEGORIES)

REFERENCE_SPLIT = "test"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"


class SampleSourceUnavailable(RuntimeError):
    """A category's source is missing, or not wired up yet.

    Separate from the built-in errors so simulate_traffic can report a missing
    corpus as a setup problem the operator can fix, rather than as a crash.
    """


class SampleLoader(Protocol):
    """Return `count` texts for one category, deterministically for a seed."""

    def __call__(self, count: int, seed: int) -> list[str]: ...


def load_in_distribution(count: int, seed: int) -> list[str]:
    """Draw held-out Gulf rows, the traffic the model was measured on.

    Reads the same test split the eval harness scores, so the reference
    distribution in a report is the one behind the documented macro F1.

    The draw is deliberately not stratified by country. The reference is meant
    to look like the split the model was scored on, class imbalance included,
    so that a shift in a report reflects the traffic rather than an artifact of
    resampling the reference into balance.
    """
    path = DEFAULT_DATA_DIR / f"gulf_{REFERENCE_SPLIT}.parquet"
    if not path.is_file():
        raise SampleSourceUnavailable(
            f"{path} not found. The splits are DVC tracked, so run "
            f"'dvc pull data/processed.dvc' first."
        )

    frame = pd.read_parquet(path)
    if "text" not in frame.columns:
        raise SampleSourceUnavailable(f"{path} has no 'text' column.")
    if count > len(frame):
        raise SampleSourceUnavailable(
            f"{path} holds {len(frame)} rows, but {count} were requested."
        )

    drawn = frame["text"].sample(n=count, random_state=seed)
    return [str(value) for value in drawn]


# Pools for the synthetic English category. The product is far larger than the
# 300 needed, so sampling without replacement returns distinct sentences with a
# spread of lengths rather than one string repeated.
_ENGLISH_SUBJECTS: tuple[str, ...] = (
    "The support team",
    "Our regional manager",
    "A new customer",
    "The billing department",
    "The night shift",
    "One of the reviewers",
    "The account holder",
    "A returning client",
    "The escalation desk",
    "The onboarding lead",
)

_ENGLISH_VERBS: tuple[str, ...] = (
    "flagged",
    "escalated",
    "reopened",
    "closed",
    "reassigned",
    "acknowledged",
    "deferred",
    "documented",
)

_ENGLISH_OBJECTS: tuple[str, ...] = (
    "the outstanding ticket",
    "this billing discrepancy",
    "the delivery complaint",
    "a duplicate account request",
    "the refund inquiry",
    "the outage report from last week",
    "the pending verification",
    "an unresolved chargeback",
)

_ENGLISH_TAILS: tuple[str, ...] = (
    "before the end of the day.",
    "and asked for a written summary.",
    "pending a second review.",
    "without contacting the customer first.",
    "after the automated check failed.",
)


def load_english(count: int, seed: int) -> list[str]:
    """Generate English sentences, which the API is expected to reject.

    Synthetic text is adequate because none of it reaches the model. Validation
    rejects anything below MIN_ARABIC_CHAR_RATIO with a 422 before the
    tokenizer, so this category measures the gate rather than a prediction
    distribution. The wording borrows the support-desk register the service is
    aimed at, so the rejected traffic resembles what a misrouted caller would
    plausibly send.
    """
    combinations = [
        f"{subject} {verb} {obj} {tail}"
        for subject in _ENGLISH_SUBJECTS
        for verb in _ENGLISH_VERBS
        for obj in _ENGLISH_OBJECTS
        for tail in _ENGLISH_TAILS
    ]
    if count > len(combinations):
        raise SampleSourceUnavailable(
            f"The English templates yield {len(combinations)} distinct "
            f"sentences, but {count} were requested."
        )
    return random.Random(seed).sample(combinations, count)


def load_msa(count: int, seed: int) -> list[str]:
    """Not wired up yet. Needs a short-form Modern Standard Arabic corpus.

    This is the sharpest of the three shifted categories and the one with no
    source in the repo. MSA is Arabic script, so it clears validation, reaches
    the model, and comes back with a confident Gulf country code that cannot be
    right. That is the drift worth measuring.
    """
    raise SampleSourceUnavailable(
        "The msa loader has no source wired up. It needs roughly 300 short "
        "Modern Standard Arabic sentences from a Wikipedia or news headline "
        "dataset. QADI cannot supply them: its 18 labels are all country "
        "dialects, with no MSA class."
    )


def load_non_gulf_dialect(count: int, seed: int) -> list[str]:
    """Not wired up yet. Needs the non-Gulf rows of the source dataset.

    The parquet splits in this repo carry the Gulf subset only, and the eval
    harness raises on any label id outside it, so these rows have to come back
    from the original dataset rather than from data/processed.
    """
    raise SampleSourceUnavailable(
        "The non_gulf_dialect loader has no source wired up. It should pull "
        "Egyptian, Levantine and Maghrebi rows at run time from the "
        "HuggingFace dataset Abdelrahman-Rezk/Arabic_Dialect_Identification. "
        "Those are source label ids outside the Gulf six, which the parquet "
        "splits in this repo do not carry."
    )


LOADERS: dict[str, SampleLoader] = {
    REFERENCE_CATEGORY: load_in_distribution,
    "msa": load_msa,
    "non_gulf_dialect": load_non_gulf_dialect,
    "english": load_english,
}


def load_category(
    category: str,
    count: int = SAMPLES_PER_CATEGORY,
    seed: int = RANDOM_SEED,
) -> list[str]:
    """Return the texts for one category.

    Raises:
        KeyError: if the category is not one of ALL_CATEGORIES.
        SampleSourceUnavailable: if its source is missing or not wired up.
    """
    if category not in LOADERS:
        raise KeyError(
            f"Unknown category {category!r}. Known categories: {sorted(LOADERS)}."
        )
    return LOADERS[category](count, seed)


def split_for(category: str) -> str:
    """Return the Evidently split a category belongs to.

    Every report compares one current category against the single reference,
    so the mapping is fixed rather than configurable.
    """
    return "reference" if category == REFERENCE_CATEGORY else "current"
