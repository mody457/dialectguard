"""Turn recorded traffic into Evidently drift reports, one per category.

Each report answers one question: when the service is fed text of some kind it
was not trained on, how far does its behavior move from the held-out Gulf
traffic the documented macro F1 was measured on. So every shifted category is
compared against the same reference rather than against each other, and each
comparison gets its own report. A single blended current set would average an
MSA shift together with an outright refusal and hide both.

What gets compared is the service's behavior, not the input text. The columns
are the prediction, its confidence, the status code and the error code. Text
drift is deliberately absent: Evidently's text descriptors pull nltk corpora at
report time, which would put a network dependency in the middle of a monitoring
run, and the interesting signal here is what the service did rather than how
the wording differed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from app.config import PROJECT_ROOT
from monitoring.samples import CURRENT_CATEGORIES

DEFAULT_TRAFFIC = PROJECT_ROOT / "monitoring" / "reports" / "traffic.parquet"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "monitoring" / "reports"

REFERENCE_SPLIT = "reference"
SERVED_STATUS = 200

# Stands in for a null in the categorical columns. A refused request has no
# dialect and a served one has no error code, so both columns are half empty by
# construction. Left as nulls the comparison would be undefined on one side.
# Filled, the shift from "every row predicted a country" to "every row was
# refused" is exactly what the report should show.
MISSING_CATEGORY = "none"

NUMERICAL_COLUMNS: tuple[str, ...] = ("confidence", "text_length")
CATEGORICAL_COLUMNS: tuple[str, ...] = ("dialect", "status_code", "error_code")

# latency_ms is recorded but not compared. It moves with whatever else the host
# is doing, so it would drift on a busy laptop and say nothing about the model.

METRICS = [DataDriftPreset()]


def load_traffic(path: Path) -> pd.DataFrame:
    """Read a simulator run, failing loudly rather than reporting on nothing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run 'python -m monitoring.simulate_traffic' "
            f"against a running instance first."
        )

    frame = pd.read_parquet(path)
    required = {"category", "split", "status_code", *NUMERICAL_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill the categorical nulls and make status_code compare as a label.

    status_code arrives as an integer, which Evidently would otherwise treat as
    a quantity and report a mean over. It is a label, so it is compared as one.
    """
    prepared = frame.copy()
    for column in ("dialect", "error_code"):
        if column in prepared.columns:
            prepared[column] = prepared[column].fillna(MISSING_CATEGORY)
    prepared["status_code"] = prepared["status_code"].astype(str)
    return prepared


def build_definition(include_predictions: bool) -> DataDefinition:
    """Describe the columns for one comparison.

    A category the service refuses outright has no served rows, so confidence
    is entirely null and dialect is entirely the fill value. Comparing those
    against a populated reference reports a drift that is really just an empty
    column. Dropping them leaves the finding that matters, which is that the
    status and error codes moved.
    """
    numerical = list(NUMERICAL_COLUMNS)
    categorical = list(CATEGORICAL_COLUMNS)
    if not include_predictions:
        numerical = [c for c in numerical if c != "confidence"]
        categorical = [c for c in categorical if c != "dialect"]
    return DataDefinition(
        numerical_columns=numerical,
        categorical_columns=categorical,
    )


def report_for_category(
    reference: pd.DataFrame, current: pd.DataFrame
) -> tuple[object, bool]:
    """Build one reference-versus-category report.

    Returns the snapshot and whether predictions were included, so the caller
    can say in its summary which comparison it actually ran.
    """
    include_predictions = (current["status_code"] == str(SERVED_STATUS)).any()
    definition = build_definition(include_predictions)

    columns = list(definition.numerical_columns or []) + list(
        definition.categorical_columns or []
    )
    reference_dataset = Dataset.from_pandas(
        reference[columns], data_definition=definition
    )
    current_dataset = Dataset.from_pandas(current[columns], data_definition=definition)

    report = Report(metrics=METRICS)
    snapshot = report.run(
        current_data=current_dataset, reference_data=reference_dataset
    )
    return snapshot, include_predictions


def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface for the report generator."""
    parser = argparse.ArgumentParser(
        description="Build one Evidently drift report per shifted category."
    )
    parser.add_argument("--traffic", type=Path, default=DEFAULT_TRAFFIC)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(CURRENT_CATEGORIES),
        choices=list(CURRENT_CATEGORIES),
        help="Categories to report on. Defaults to all three shifted sets.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate the reports and return a process exit code."""
    args = build_parser().parse_args(argv)

    frame = prepare(load_traffic(args.traffic))
    reference = frame[frame["split"] == REFERENCE_SPLIT]
    if reference.empty:
        print(f"{args.traffic} holds no reference rows to compare against.")
        return 1

    args.report_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for category in args.categories:
        current = frame[frame["category"] == category]
        if current.empty:
            print(f"Skipping {category}: no rows in {args.traffic}.")
            continue

        snapshot, include_predictions = report_for_category(reference, current)
        html_path = args.report_dir / f"drift_{category}.html"
        snapshot.save_html(str(html_path))
        snapshot.save_json(str(args.report_dir / f"drift_{category}.json"))
        written += 1

        compared = (
            "behavior and predictions" if include_predictions else "behavior only"
        )
        print(
            f"{category}: {len(current)} rows against {len(reference)} reference, "
            f"{compared}, written to {html_path}"
        )

    if not written:
        print("No category produced a report.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
