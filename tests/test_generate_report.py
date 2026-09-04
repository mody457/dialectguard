"""Tests for the drift report generator.

These run Evidently for real on a small synthetic frame rather than stubbing
it. The wiring is the whole risk here: the column roles, and what happens to a
category the service refused outright, where confidence is entirely null. A
stubbed Evidently would prove none of that.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from monitoring import generate_report

REFERENCE_ROWS = 40
CURRENT_ROWS = 30


def served_rows(category: str, split: str, count: int) -> list[dict[str, object]]:
    """Rows that got a prediction back."""
    dialects = ["KW", "SA", "QA", "AE", "BH", "OM"]
    return [
        {
            "category": category,
            "split": split,
            "text": f"row {index}",
            "text_length": 20 + index % 15,
            "status_code": 200,
            "dialect": dialects[index % len(dialects)],
            "confidence": 0.5 + (index % 40) / 100,
            "error_code": None,
            "latency_ms": 12.0 + index % 5,
        }
        for index in range(count)
    ]


def refused_rows(category: str, count: int) -> list[dict[str, object]]:
    """Rows the validation gate rejected, so no dialect and no confidence."""
    return [
        {
            "category": category,
            "split": "current",
            "text": f"english row {index}",
            "text_length": 55 + index % 20,
            "status_code": 422,
            "dialect": None,
            "confidence": None,
            "error_code": "NOT_ARABIC_DOMINANT",
            "latency_ms": 3.0 + index % 3,
        }
        for index in range(count)
    ]


def write_traffic(path: Path) -> Path:
    """A run holding a reference, one served category and one refused one."""
    rows = (
        served_rows("in_distribution", "reference", REFERENCE_ROWS)
        + served_rows("msa", "current", CURRENT_ROWS)
        + refused_rows("english", CURRENT_ROWS)
    )
    target = path / "traffic.parquet"
    pd.DataFrame(rows).to_parquet(target, index=False)
    return target


def test_load_traffic_points_at_the_simulator_when_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="simulate_traffic"):
        generate_report.load_traffic(tmp_path / "traffic.parquet")


def test_load_traffic_rejects_a_frame_without_the_required_columns(
    tmp_path: Path,
) -> None:
    target = tmp_path / "traffic.parquet"
    pd.DataFrame({"category": ["msa"]}).to_parquet(target, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        generate_report.load_traffic(target)


def test_prepare_fills_the_categorical_nulls(tmp_path: Path) -> None:
    """Nulls on one side would make the categorical comparison undefined."""
    frame = pd.DataFrame(refused_rows("english", 3))
    prepared = generate_report.prepare(frame)
    assert (prepared["dialect"] == generate_report.MISSING_CATEGORY).all()
    assert prepared["error_code"].notna().all()


def test_prepare_compares_status_code_as_a_label(tmp_path: Path) -> None:
    """Left as an integer, Evidently would report a mean HTTP status."""
    frame = pd.DataFrame(served_rows("msa", "current", 3))
    prepared = generate_report.prepare(frame)["status_code"]
    assert pd.api.types.is_string_dtype(prepared)
    assert list(prepared) == ["200", "200", "200"]


def test_build_definition_drops_predictions_when_nothing_was_served() -> None:
    definition = generate_report.build_definition(include_predictions=False)
    assert "confidence" not in (definition.numerical_columns or [])
    assert "dialect" not in (definition.categorical_columns or [])
    assert "status_code" in (definition.categorical_columns or [])


def test_build_definition_keeps_predictions_when_rows_were_served() -> None:
    definition = generate_report.build_definition(include_predictions=True)
    assert "confidence" in (definition.numerical_columns or [])
    assert "dialect" in (definition.categorical_columns or [])


def test_main_writes_a_report_per_category(tmp_path: Path) -> None:
    """The end to end path, with Evidently actually running."""
    traffic = write_traffic(tmp_path)
    report_dir = tmp_path / "out"

    exit_code = generate_report.main(
        [
            "--traffic",
            str(traffic),
            "--report-dir",
            str(report_dir),
            "--categories",
            "msa",
            "english",
        ]
    )

    assert exit_code == 0
    for category in ("msa", "english"):
        assert (report_dir / f"drift_{category}.html").is_file()
        assert (report_dir / f"drift_{category}.json").is_file()


def test_main_reports_on_a_refused_category_without_predictions(
    tmp_path: Path,
) -> None:
    """English has no served rows, which must not stop its report."""
    traffic = write_traffic(tmp_path)
    report_dir = tmp_path / "out"

    exit_code = generate_report.main(
        [
            "--traffic",
            str(traffic),
            "--report-dir",
            str(report_dir),
            "--categories",
            "english",
        ]
    )

    assert exit_code == 0
    assert (report_dir / "drift_english.html").stat().st_size > 0


def test_main_fails_when_there_is_no_reference(tmp_path: Path) -> None:
    """Reporting against an empty reference would compare against nothing."""
    target = tmp_path / "traffic.parquet"
    pd.DataFrame(refused_rows("english", 5)).to_parquet(target, index=False)

    exit_code = generate_report.main(
        ["--traffic", str(target), "--report-dir", str(tmp_path / "out")]
    )
    assert exit_code == 1
