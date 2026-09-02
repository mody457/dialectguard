"""Tests for the evaluation harness.

None of these load the checkpoint. Scoring the real split takes minutes, which
does not belong in the unit suite, so the model call is stubbed and what gets
tested is the wiring around it: the label mapping, the row accounting and the
threshold gate. Those are the parts that fail silently. A broken label mapping
would still produce a plausible-looking number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.config import ID_TO_DIALECT
from eval_harness import evaluate

# Source dataset ids for OM, SA, KW, QA, BH, AE, in model label order.
EXPECTED_SOURCE_IDS = [0, 2, 3, 4, 13, 15]


def test_source_to_model_labels_matches_the_training_notebook() -> None:
    """The mapping must reproduce old_to_new from the training notebook."""
    mapping = evaluate.source_to_model_labels()
    assert mapping == dict(
        zip(EXPECTED_SOURCE_IDS, range(len(ID_TO_DIALECT)), strict=True)
    )


def test_source_to_model_labels_rejects_an_unknown_dialect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dialect absent from the source labels must raise, not be skipped."""
    monkeypatch.setattr(evaluate, "ID_TO_DIALECT", ("OM", "ZZ"))
    with pytest.raises(ValueError, match="ZZ"):
        evaluate.source_to_model_labels()


def write_parquet(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_load_split_reports_a_missing_file_usefully(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        evaluate.load_split(tmp_path, "test")


def test_load_split_rejects_a_frame_without_the_required_columns(
    tmp_path: Path,
) -> None:
    write_parquet(tmp_path / "gulf_test.parquet", [{"id": 1, "label": 0}])
    with pytest.raises(ValueError, match="text"):
        evaluate.load_split(tmp_path, "test")


def test_load_split_honours_the_row_limit(tmp_path: Path) -> None:
    write_parquet(
        tmp_path / "gulf_test.parquet",
        [{"id": i, "label": 0, "text": "شلونك"} for i in range(10)],
    )
    assert len(evaluate.load_split(tmp_path, "test", limit=4)) == 4


def test_prepare_maps_labels_and_cleans_text() -> None:
    frame = pd.DataFrame(
        [
            {"label": 0, "text": "@ali شلونك"},
            {"label": 15, "text": "شخبارك #دبي"},
        ]
    )
    texts, labels, empty = evaluate.prepare(frame)

    assert texts == ["شلونك", "شخبارك"]
    # Source id 0 is OM (model 0) and 15 is AE (model 5).
    assert labels.tolist() == [0, 5]
    assert empty == 0


def test_prepare_counts_rows_that_clean_to_nothing_without_dropping_them() -> None:
    """Empty rows are counted but kept, so the denominator stays honest."""
    frame = pd.DataFrame(
        [
            {"label": 0, "text": "@user #tag https://example.com"},
            {"label": 3, "text": "شلونك"},
        ]
    )
    texts, labels, empty = evaluate.prepare(frame)

    assert empty == 1
    assert len(texts) == 2
    assert len(labels) == 2


def test_prepare_rejects_labels_outside_the_gulf_subset() -> None:
    """A non-Gulf id means the wrong split was handed in. Fail, do not filter."""
    frame = pd.DataFrame([{"label": 10, "text": "شلونك"}])
    with pytest.raises(ValueError, match="outside the Gulf subset"):
        evaluate.prepare(frame)


def test_score_computes_macro_f1_and_support() -> None:
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2])
    result = evaluate.score("test", y_true, y_pred, empty_after_cleaning=0)

    assert result.rows == 4
    assert result.accuracy == pytest.approx(0.75)
    assert set(result.per_class_f1) == set(ID_TO_DIALECT)
    assert result.support["OM"] == 2
    assert result.macro_f1 == pytest.approx(
        float(np.mean([2 / 3, 2 / 3, 1.0, 0.0, 0.0, 0.0]))
    )
    assert len(result.confusion) == len(ID_TO_DIALECT)


def test_confusion_as_markdown_has_a_row_per_dialect() -> None:
    result = evaluate.score(
        "test", np.array([0, 1]), np.array([0, 1]), empty_after_cleaning=0
    )
    lines = evaluate.confusion_as_markdown(result).splitlines()
    # Header, divider, then one row per dialect.
    assert len(lines) == len(ID_TO_DIALECT) + 2
    for code in ID_TO_DIALECT:
        assert f"**{code}**" in "\n".join(lines)


def stub_split(tmp_path: Path) -> Path:
    """Two rows per dialect, so a perfect prediction gives macro F1 1.0."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": i, "label": source_id, "text": "شلونك"}
        for i, source_id in enumerate(EXPECTED_SOURCE_IDS * 2)
    ]
    write_parquet(tmp_path / "gulf_test.parquet", rows)
    return tmp_path


def test_main_passes_the_gate_on_perfect_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = stub_split(tmp_path / "data")
    monkeypatch.setattr(
        evaluate,
        "predict_labels",
        lambda texts, model_dir, batch_size: np.array(
            list(range(len(ID_TO_DIALECT))) * 2
        ),
    )
    exit_code = evaluate.main(
        [
            "--data-dir", str(data_dir),
            "--report-dir", str(tmp_path / "reports"),
            "--no-plot",
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "reports" / "eval_report.json").is_file()
    assert (tmp_path / "reports" / "confusion_matrix.md").is_file()


def test_main_fails_the_gate_when_macro_f1_is_below_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is what CI depends on, so a bad model must exit non-zero."""
    data_dir = stub_split(tmp_path / "data")
    monkeypatch.setattr(
        evaluate,
        "predict_labels",
        lambda texts, model_dir, batch_size: np.zeros(
            len(ID_TO_DIALECT) * 2, dtype=int
        ),
    )
    exit_code = evaluate.main(
        [
            "--data-dir", str(data_dir),
            "--report-dir", str(tmp_path / "reports"),
            "--no-plot",
        ]
    )
    assert exit_code == 1
