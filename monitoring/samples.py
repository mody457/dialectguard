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
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from app.config import ID_TO_DIALECT, PROJECT_ROOT, SOURCE_LABEL_NAMES

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

HUB_REPO_TYPE = "dataset"

# The Gulf splits under data/processed are the Gulf subset of this dataset, so
# its test split is the exact complement of the reference: same corpus, same
# collection, same split, only the dialect differs. That is what makes the
# comparison a dialect shift rather than a change of source.
NON_GULF_REPO_ID = "Abdelrahman-Rezk/Arabic_Dialect_Identification"
NON_GULF_SPLIT = "test"

# Egyptian, Levantine and Maghrebi. Iraqi, Yemeni and Sudanese are left out
# deliberately. They border the Gulf and share features with it, so a Gulf
# label on one of them is not clearly wrong, and this category is only useful
# if every row in it is Arabic the model has no right answer for.
NON_GULF_DIALECTS: tuple[str, ...] = (
    "EG",
    "LB", "JO", "SY", "PL",
    "MA", "DZ", "TN", "LY",
)

# BBC Arabic, taken as headlines rather than article bodies. Headlines are one
# edited sentence of Modern Standard Arabic, which keeps this category's length
# distribution close to the tweet reference (median 55 characters against 71).
# Article bodies average around 2500 characters, so they would be rejected on
# length and the report would show a body-size shift instead of a dialect one.
MSA_REPO_ID = "Abdelkareem/arabic-bbc-news"
MSA_SPLIT = "train"
MSA_TEXT_COLUMN = "title"


class SampleSourceUnavailable(RuntimeError):
    """A category's source is missing, or not wired up yet.

    Separate from the built-in errors so simulate_traffic can report a missing
    corpus as a setup problem the operator can fix, rather than as a crash.
    """


class SampleLoader(Protocol):
    """Return `count` texts for one category, deterministically for a seed."""

    def __call__(self, count: int, seed: int) -> list[str]: ...


def fetch_hub_split(repo_id: str, split: str) -> pd.DataFrame:
    """Download one split of a Hub dataset and return it as a frame.

    The split's parquet file is resolved from the repo listing rather than
    named outright, because one of these repos suffixes its filenames with a
    content hash that changes whenever the data is re-uploaded.

    Only the file backing the requested split is fetched. Reading these through
    datasets.load_dataset would pull every split in the repo, which for the
    dialect corpus means 48MB of training rows to reach a 1MB test split, and
    would add a dependency for work huggingface-hub already does.

    Raises:
        SampleSourceUnavailable: if the repo, the split or the network is not
            there. These sources are fetched on demand, so an unreachable Hub
            is a setup problem for the operator rather than a bug.
    """
    try:
        listing = HfApi().list_repo_files(repo_id, repo_type=HUB_REPO_TYPE)
        candidates = sorted(
            name
            for name in listing
            if name.startswith(f"data/{split}-") and name.endswith(".parquet")
        )
        if not candidates:
            raise SampleSourceUnavailable(
                f"{repo_id} has no parquet file for the {split!r} split."
            )
        frames = [
            pd.read_parquet(
                hf_hub_download(repo_id, name, repo_type=HUB_REPO_TYPE)
            )
            for name in candidates
        ]
    except (OSError, EntryNotFoundError) as exc:
        raise SampleSourceUnavailable(
            f"Could not read the {split!r} split of {repo_id} from the "
            f"HuggingFace Hub. These sources are downloaded on demand, so "
            f"this needs a working network connection. ({exc})"
        ) from exc

    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def draw_distinct(texts: pd.Series, count: int, seed: int, source: str) -> list[str]:
    """Sample `count` distinct non-empty texts, or say why it could not.

    Duplicates are dropped before sampling rather than after, so the result is
    always the requested size. A report built on the same sentence repeated
    would show a confidence distribution far tighter than real traffic.
    """
    cleaned = texts.dropna().astype(str).str.strip()
    usable = cleaned[cleaned.str.len() > 0].drop_duplicates()
    if count > len(usable):
        raise SampleSourceUnavailable(
            f"{source} yields {len(usable)} distinct texts, but {count} were "
            f"requested."
        )
    return [str(value) for value in usable.sample(n=count, random_state=seed)]


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
    """Draw Modern Standard Arabic headlines, the sharpest of the three shifts.

    MSA is Arabic script, so it clears validation, reaches the model, and comes
    back with a confident Gulf country code that cannot be right. Nothing in
    the pipeline can notice: there is no MSA label to predict and no signal in
    the response that the input was out of domain. That is the drift worth
    measuring, and the reason this category matters more than the English one,
    which at least gets refused.

    The source dataset is not the one the model was trained on, because none
    exists. QADI's 18 labels are all country dialects with no MSA class, so
    there is no way to hold the corpus fixed the way non_gulf_dialect does.
    That leaves a confound worth stating: news headlines differ from tweets in
    register and subject as well as in variety, so a shift here is MSA news
    writing against Gulf tweets, not MSA against dialect in the abstract.
    """
    frame = fetch_hub_split(MSA_REPO_ID, MSA_SPLIT)
    if MSA_TEXT_COLUMN not in frame.columns:
        raise SampleSourceUnavailable(
            f"{MSA_REPO_ID} has no {MSA_TEXT_COLUMN!r} column."
        )
    return draw_distinct(
        frame[MSA_TEXT_COLUMN], count, seed, f"{MSA_REPO_ID} {MSA_TEXT_COLUMN}s"
    )


def non_gulf_label_ids() -> dict[int, str]:
    """Map the source label ids this category draws from to their codes.

    Built from SOURCE_LABEL_NAMES so the ids are never written down here, and
    checked against ID_TO_DIALECT so a dialect the model actually predicts can
    never end up in the shifted set. That check is the point: if the two ever
    overlap, the report would compare Gulf traffic against Gulf traffic and
    show no drift, which reads as a passing monitor rather than a broken one.

    Raises:
        ValueError: if a selected dialect is unknown to the source dataset or
            is one of the six the model emits.
    """
    ids: dict[int, str] = {}
    for code in NON_GULF_DIALECTS:
        if code not in SOURCE_LABEL_NAMES:
            raise ValueError(f"{code!r} is not a source dataset label.")
        if code in ID_TO_DIALECT:
            raise ValueError(f"{code!r} is one of the six dialects the model predicts.")
        ids[SOURCE_LABEL_NAMES.index(code)] = code
    return ids


def load_non_gulf_dialect(count: int, seed: int) -> list[str]:
    """Draw Egyptian, Levantine and Maghrebi rows: Arabic with no right label.

    These come from the test split of the dataset the Gulf splits under
    data/processed were carved out of, so the only thing separating this
    category from the reference is the dialect. Same corpus, same collection,
    same split. A shift in the report is therefore about the text rather than
    about switching sources, which is not something a corpus scraped somewhere
    else could support.

    Unlike the reference, the draw is stratified evenly across the dialects.
    The reference keeps its class imbalance because that is the distribution
    the documented macro F1 was measured on. This category has no such
    distribution to honour: its per-dialect counts are an artifact of how much
    each country tweets, and left alone the draw would be mostly Egyptian and
    would really be measuring one dialect rather than nine.
    """
    frame = fetch_hub_split(NON_GULF_REPO_ID, NON_GULF_SPLIT)
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise SampleSourceUnavailable(
            f"{NON_GULF_REPO_ID} is missing columns: {sorted(missing)}."
        )

    wanted = non_gulf_label_ids()
    per_dialect, remainder = divmod(count, len(wanted))

    texts: list[str] = []
    for position, (label_id, code) in enumerate(sorted(wanted.items())):
        # The remainder goes to the first few dialects in label id order, so an
        # uneven count still splits the same way on every run.
        quota = per_dialect + (1 if position < remainder else 0)
        if not quota:
            continue
        rows = frame.loc[frame["label"] == label_id, "text"]
        texts.extend(
            draw_distinct(
                rows, quota, seed, f"{NON_GULF_REPO_ID} {code} rows"
            )
        )

    # Interleave the dialects so request order does not run country by country.
    random.Random(seed).shuffle(texts)
    return texts


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
