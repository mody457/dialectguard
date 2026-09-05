"""Tests for the drift simulation's sample sources.

The loaders are the part of the monitoring path that fails quietly. A source
that returned 280 rows instead of 300, or that reshuffled between runs, would
still produce a report, just one that says the wrong thing. These check the
counts, the determinism and that an unwired source names what it needs instead
of raising something opaque half way through a run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from monitoring import samples


def write_split(path: Path, rows: int) -> None:
    """Write a stand-in for gulf_test.parquet, ASCII so the console is safe."""
    gulf_ids = [0, 2, 3, 4, 13, 15]
    frame = pd.DataFrame(
        {
            "id": range(rows),
            "label": [gulf_ids[index % len(gulf_ids)] for index in range(rows)],
            "text": [f"row {index}" for index in range(rows)],
        }
    )
    frame.to_parquet(path / "gulf_test.parquet", index=False)


def test_every_category_has_a_registered_loader() -> None:
    """ALL_CATEGORIES and the registry must not drift apart."""
    assert set(samples.ALL_CATEGORIES) == set(samples.LOADERS)


def test_reference_is_not_also_a_current_category() -> None:
    """The reference has to sit outside the shifted sets it is compared to."""
    assert samples.REFERENCE_CATEGORY not in samples.CURRENT_CATEGORIES


def test_split_for_puts_only_the_reference_in_reference() -> None:
    assert samples.split_for(samples.REFERENCE_CATEGORY) == "reference"
    for category in samples.CURRENT_CATEGORIES:
        assert samples.split_for(category) == "current"


def test_load_english_returns_the_requested_count_all_distinct() -> None:
    """Sampling without replacement, so a report is not built on repeats."""
    texts = samples.load_english(samples.SAMPLES_PER_CATEGORY, samples.RANDOM_SEED)
    assert len(texts) == samples.SAMPLES_PER_CATEGORY
    assert len(set(texts)) == samples.SAMPLES_PER_CATEGORY


def test_load_english_is_deterministic_for_a_seed() -> None:
    """Two runs at the same seed have to be comparable to each other."""
    assert samples.load_english(50, 7) == samples.load_english(50, 7)


def test_load_english_varies_with_the_seed() -> None:
    assert samples.load_english(50, 7) != samples.load_english(50, 8)


def test_load_english_carries_no_arabic() -> None:
    """The category exists to be refused, so it must fail the ratio gate."""
    from app.preprocessing import arabic_ratio

    for text in samples.load_english(20, samples.RANDOM_SEED):
        assert arabic_ratio(text) == 0.0


def test_load_english_refuses_more_than_the_templates_can_make() -> None:
    with pytest.raises(samples.SampleSourceUnavailable, match="distinct"):
        samples.load_english(10_000, samples.RANDOM_SEED)


def test_load_in_distribution_points_at_dvc_when_the_split_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh checkout has no parquet. The message has to say what to run."""
    monkeypatch.setattr(samples, "DEFAULT_DATA_DIR", tmp_path)
    with pytest.raises(samples.SampleSourceUnavailable, match="dvc pull"):
        samples.load_in_distribution(10, samples.RANDOM_SEED)


def test_load_in_distribution_rejects_a_split_without_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(samples, "DEFAULT_DATA_DIR", tmp_path)
    pd.DataFrame({"id": [1], "label": [0]}).to_parquet(
        tmp_path / "gulf_test.parquet", index=False
    )
    with pytest.raises(samples.SampleSourceUnavailable, match="text"):
        samples.load_in_distribution(1, samples.RANDOM_SEED)


def test_load_in_distribution_refuses_to_oversample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asking for more rows than exist must raise, not silently return fewer."""
    monkeypatch.setattr(samples, "DEFAULT_DATA_DIR", tmp_path)
    write_split(tmp_path, 20)
    with pytest.raises(samples.SampleSourceUnavailable, match="20 rows"):
        samples.load_in_distribution(50, samples.RANDOM_SEED)


def test_load_in_distribution_is_deterministic_for_a_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(samples, "DEFAULT_DATA_DIR", tmp_path)
    write_split(tmp_path, 100)
    first = samples.load_in_distribution(30, samples.RANDOM_SEED)
    second = samples.load_in_distribution(30, samples.RANDOM_SEED)
    assert first == second
    assert len(first) == 30


def test_unwired_loaders_name_what_they_need() -> None:
    """An unwired source is a setup problem, so the message has to be usable."""
    with pytest.raises(samples.SampleSourceUnavailable) as excinfo:
        samples.load_category("msa", 10, samples.RANDOM_SEED)
    assert "no source wired up" in str(excinfo.value)


def test_load_msa_explains_why_qadi_cannot_supply_it() -> None:
    """The obvious fix is wrong, so the message has to rule it out."""
    with pytest.raises(samples.SampleSourceUnavailable, match="no MSA class"):
        samples.load_msa(10, samples.RANDOM_SEED)


def test_load_category_rejects_an_unknown_category() -> None:
    with pytest.raises(KeyError, match="gulf_arabic"):
        samples.load_category("gulf_arabic", 10, samples.RANDOM_SEED)


def non_gulf_frame(rows_per_dialect: dict[str, int]) -> pd.DataFrame:
    """Build a stand-in for the Hub split, ASCII so the console is safe.

    Texts carry their dialect code so a test can tell which rows a draw took.
    """
    records: list[dict[str, object]] = []
    for code, rows in rows_per_dialect.items():
        label_id = samples.SOURCE_LABEL_NAMES.index(code)
        for index in range(rows):
            records.append(
                {"id": len(records), "label": label_id, "text": f"{code} row {index}"}
            )
    return pd.DataFrame(records)


def even_non_gulf_frame(rows_per_dialect: int = 50) -> pd.DataFrame:
    return non_gulf_frame(dict.fromkeys(samples.NON_GULF_DIALECTS, rows_per_dialect))


def test_non_gulf_label_ids_resolve_through_the_source_label_order() -> None:
    """Ids are derived, never written down, so this pins what they resolve to."""
    assert samples.non_gulf_label_ids() == {
        10: "EG",
        5: "LB",
        6: "JO",
        7: "SY",
        11: "PL",
        9: "MA",
        14: "DZ",
        16: "TN",
        17: "LY",
    }


def test_non_gulf_dialects_never_overlap_the_predicted_six() -> None:
    """An overlap would compare Gulf against Gulf and report no drift."""
    assert not set(samples.NON_GULF_DIALECTS) & set(samples.ID_TO_DIALECT)


def test_non_gulf_label_ids_rejects_a_dialect_the_source_does_not_have(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(samples, "NON_GULF_DIALECTS", ("EG", "ZZ"))
    with pytest.raises(ValueError, match="ZZ"):
        samples.non_gulf_label_ids()


def test_non_gulf_label_ids_rejects_a_dialect_the_model_predicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard that keeps the shifted set genuinely shifted."""
    monkeypatch.setattr(samples, "NON_GULF_DIALECTS", ("EG", "KW"))
    with pytest.raises(ValueError, match="KW"):
        samples.non_gulf_label_ids()


def test_load_non_gulf_dialect_returns_the_count_all_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        samples, "fetch_hub_split", lambda repo, split: even_non_gulf_frame()
    )
    texts = samples.load_non_gulf_dialect(90, samples.RANDOM_SEED)
    assert len(texts) == 90
    assert len(set(texts)) == 90


def test_load_non_gulf_dialect_draws_evenly_despite_a_skewed_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real split is mostly Egyptian. An unstratified draw would be too."""
    skewed = non_gulf_frame(
        {"EG": 400, "LB": 40, "JO": 40, "SY": 40, "PL": 40,
         "MA": 40, "DZ": 40, "TN": 40, "LY": 40}
    )
    monkeypatch.setattr(samples, "fetch_hub_split", lambda repo, split: skewed)
    texts = samples.load_non_gulf_dialect(90, samples.RANDOM_SEED)
    per_dialect = pd.Series([text.split()[0] for text in texts]).value_counts()
    assert set(per_dialect.index) == set(samples.NON_GULF_DIALECTS)
    assert per_dialect.unique().tolist() == [10]


def test_load_non_gulf_dialect_spreads_an_uneven_count_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """92 over 9 dialects cannot be even, but it must split the same way twice."""
    monkeypatch.setattr(
        samples, "fetch_hub_split", lambda repo, split: even_non_gulf_frame()
    )
    first = samples.load_non_gulf_dialect(92, samples.RANDOM_SEED)
    assert len(first) == 92
    assert first == samples.load_non_gulf_dialect(92, samples.RANDOM_SEED)


def test_load_non_gulf_dialect_is_deterministic_for_a_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        samples, "fetch_hub_split", lambda repo, split: even_non_gulf_frame()
    )
    first = samples.load_non_gulf_dialect(45, 7)
    assert first == samples.load_non_gulf_dialect(45, 7)
    assert first != samples.load_non_gulf_dialect(45, 8)


def test_load_non_gulf_dialect_does_not_order_the_run_by_dialect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sending country by country would confound dialect with request order."""
    monkeypatch.setattr(
        samples, "fetch_hub_split", lambda repo, split: even_non_gulf_frame()
    )
    codes = [text.split()[0] for text in samples.load_non_gulf_dialect(90, 42)]
    assert codes != sorted(codes)


def test_load_non_gulf_dialect_names_the_dialect_it_ran_short_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A thin dialect has to be identifiable, not just a count that came up low."""
    thin = non_gulf_frame(
        {"EG": 50, "LB": 50, "JO": 50, "SY": 50, "PL": 50,
         "MA": 50, "DZ": 50, "TN": 50, "LY": 2}
    )
    monkeypatch.setattr(samples, "fetch_hub_split", lambda repo, split: thin)
    with pytest.raises(samples.SampleSourceUnavailable, match="LY"):
        samples.load_non_gulf_dialect(90, samples.RANDOM_SEED)


def test_load_non_gulf_dialect_rejects_a_split_without_the_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        samples,
        "fetch_hub_split",
        lambda repo, split: pd.DataFrame({"id": [1], "label": [10]}),
    )
    with pytest.raises(samples.SampleSourceUnavailable, match="text"):
        samples.load_non_gulf_dialect(10, samples.RANDOM_SEED)


def test_draw_distinct_drops_duplicates_and_blanks() -> None:
    """A report built on a repeated sentence would understate the spread."""
    series = pd.Series(["one", "one", "  ", "two", None, "three", "three"])
    drawn = samples.draw_distinct(series, 3, samples.RANDOM_SEED, "stub")
    assert sorted(drawn) == ["one", "three", "two"]


def test_draw_distinct_counts_distinct_rows_not_raw_rows() -> None:
    """Four rows but two texts, so asking for three has to raise."""
    series = pd.Series(["one", "one", "two", "two"])
    with pytest.raises(samples.SampleSourceUnavailable, match="2 distinct"):
        samples.draw_distinct(series, 3, samples.RANDOM_SEED, "stub")
