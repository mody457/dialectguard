"""Score the shipped checkpoint on the held-out Gulf test split.

CI runs this as a gate before any Docker build. It exists for two reasons.

The obvious one is regression detection. A swapped checkpoint, a reordered
label map or a weakened preprocessing step all show up here as a drop in macro
F1, and none of them would fail a unit test.

The less obvious one is train/serve parity. The cleaning code used by the clean
retrain was never saved, so it cannot be diffed against app.preprocessing.
Running the raw test split through app.preprocessing and landing at the
documented macro F1 is the evidence that the two agree. A large gap means they
do not, which is a train/serve skew nothing else in the suite would catch.

One deliberate difference from the serving path: training tokenized with
truncation=True, so examples over the token budget were clipped. The API
rejects them instead, which is right for a caller who would otherwise get a
confident answer about half a sentence. This harness follows training rather
than serving, because it measures the checkpoint against the number training
produced. Rejecting here would quietly change the denominator.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import ID_TO_DIALECT, MAX_SEQUENCE_LENGTH, MODEL_DIR, PROJECT_ROOT
from app.preprocessing import preprocess

# The 18 dialect labels of the source dataset, in the order its dataset card
# lists them. The parquet splits carry these ids, so they have to be mapped down
# to the six the model emits before anything can be compared.
SOURCE_LABEL_NAMES: tuple[str, ...] = (
    "OM", "SD", "SA", "KW", "QA", "LB", "JO", "SY", "IQ",
    "MA", "EG", "PL", "YE", "BH", "DZ", "AE", "TN", "LY",
)

# Documented macro F1 is 0.618. The gate sits below it so library drift cannot
# fail a build over noise, while leaving no room for a real regression: a wrong
# label order or a dropped preprocessing step costs far more than two points.
DEFAULT_MIN_MACRO_F1 = 0.60

DEFAULT_BATCH_SIZE = 32
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "docs"


@dataclass(frozen=True)
class EvalResult:
    """Everything the gate needs to decide, and a human needs to diagnose."""

    split: str
    rows: int
    macro_f1: float
    accuracy: float
    per_class_f1: dict[str, float]
    support: dict[str, int]
    confusion: list[list[int]]
    empty_after_cleaning: int


def source_to_model_labels() -> dict[int, int]:
    """Map the source dataset's 18 label ids onto the model's six.

    Mirrors old_to_new from the training notebook, but is built from
    ID_TO_DIALECT so the serving label order stays the single source of truth.
    If the two ever disagree this raises, rather than quietly scoring
    predictions against the wrong countries.
    """
    mapping: dict[int, int] = {}
    for model_index, code in enumerate(ID_TO_DIALECT):
        if code not in SOURCE_LABEL_NAMES:
            raise ValueError(
                f"Dialect {code!r} from ID_TO_DIALECT is not a source dataset label."
            )
        mapping[SOURCE_LABEL_NAMES.index(code)] = model_index
    return mapping


def load_split(data_dir: Path, split: str, limit: int | None = None) -> pd.DataFrame:
    """Read one parquet split, failing loudly on a missing file or column."""
    path = data_dir / f"gulf_{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Split not found: {path}. The data is DVC tracked, so run "
            f"dvc pull first."
        )
    frame = pd.read_parquet(path)
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame.head(limit) if limit else frame


def prepare(frame: pd.DataFrame) -> tuple[list[str], np.ndarray, int]:
    """Clean the text and map gold labels, without dropping any rows.

    Rows that clean down to nothing are counted and kept. Dropping them would
    change the denominator and flatter the score against the documented number.
    """
    mapping = source_to_model_labels()
    unknown = sorted(set(frame["label"].tolist()) - set(mapping))
    if unknown:
        raise ValueError(
            f"Split carries label ids outside the Gulf subset: {unknown}. "
            f"Expected only {sorted(mapping)}."
        )

    labels = frame["label"].map(mapping).to_numpy()
    texts = [preprocess(str(value)) for value in frame["text"]]
    empty_after_cleaning = sum(1 for text in texts if not text)
    return texts, labels, empty_after_cleaning


@torch.inference_mode()
def predict_labels(
    texts: list[str], model_dir: Path, batch_size: int = DEFAULT_BATCH_SIZE
) -> np.ndarray:
    """Run the checkpoint over every text and return predicted class indices.

    Batches are padded to the longest member rather than to a fixed 128. The
    attention mask makes the two equivalent, and the measured difference in
    output probability is float noise (see README), so the faster form is used.
    """
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    predictions: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        )
        logits = model(**encoded).logits
        predictions.append(torch.argmax(logits, dim=-1).numpy())
    return np.concatenate(predictions)


def score(
    split: str, y_true: np.ndarray, y_pred: np.ndarray, empty_after_cleaning: int
) -> EvalResult:
    """Compute the gate metric and the per-country detail behind it."""
    indices = list(range(len(ID_TO_DIALECT)))
    per_class = f1_score(y_true, y_pred, average=None, labels=indices, zero_division=0)
    # The macro average is taken from per_class rather than asked of sklearn a
    # second time. Without an explicit labels argument sklearn averages only
    # over the classes present in the data, so a split or a --limit slice that
    # happened to omit a dialect would report an inflated gate metric that
    # disagreed with the per-class breakdown printed beside it.
    return EvalResult(
        split=split,
        rows=len(y_true),
        macro_f1=float(np.mean(per_class)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        per_class_f1={
            code: float(value)
            for code, value in zip(ID_TO_DIALECT, per_class, strict=True)
        },
        support={
            code: int((y_true == index).sum())
            for index, code in enumerate(ID_TO_DIALECT)
        },
        confusion=confusion_matrix(y_true, y_pred, labels=indices).tolist(),
        empty_after_cleaning=empty_after_cleaning,
    )


def confusion_as_markdown(result: EvalResult) -> str:
    """Render the confusion matrix as a table, so it is diffable in review."""
    header = "| true / pred | " + " | ".join(ID_TO_DIALECT) + " |"
    divider = "|---" * (len(ID_TO_DIALECT) + 1) + "|"
    rows = [
        f"| **{code}** | " + " | ".join(str(count) for count in row) + " |"
        for code, row in zip(ID_TO_DIALECT, result.confusion, strict=True)
    ]
    return "\n".join([header, divider, *rows])


def write_reports(result: EvalResult, report_dir: Path, plot: bool = True) -> None:
    """Persist the metrics, the matrix table and, optionally, the heatmap."""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "eval_report.json").write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "confusion_matrix.md").write_text(
        f"# Confusion matrix ({result.split} split)\n\n"
        f"Rows are the true dialect, columns the predicted one. "
        f"Macro F1 {result.macro_f1:.3f} over {result.rows} examples.\n\n"
        + confusion_as_markdown(result)
        + "\n",
        encoding="utf-8",
    )
    if not plot:
        return

    import matplotlib

    matplotlib.use("Agg")  # No display in CI, and none needed to write a file.
    import matplotlib.pyplot as plt

    matrix = np.array(result.confusion)
    figure, axes = plt.subplots(figsize=(7, 6))
    axes.imshow(matrix, cmap="Blues")
    axes.set_xticks(range(len(ID_TO_DIALECT)), ID_TO_DIALECT)
    axes.set_yticks(range(len(ID_TO_DIALECT)), ID_TO_DIALECT)
    axes.set_xlabel("Predicted dialect")
    axes.set_ylabel("True dialect")
    axes.set_title(f"Gulf dialect confusion, macro F1 {result.macro_f1:.3f}")
    midpoint = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > midpoint else "black",
            )
    figure.tight_layout()
    figure.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface for the harness."""
    parser = argparse.ArgumentParser(description="Score the checkpoint on a split.")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--min-macro-f1",
        type=float,
        default=DEFAULT_MIN_MACRO_F1,
        help="Gate threshold. Exit code is 1 below this.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Score only the first N rows."
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip the confusion heatmap."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and return a process exit code."""
    args = build_parser().parse_args(argv)

    frame = load_split(args.data_dir, args.split, args.limit)
    texts, y_true, empty_after_cleaning = prepare(frame)
    print(f"Scoring {len(texts)} rows from the {args.split} split.")
    if empty_after_cleaning:
        print(
            f"Note: {empty_after_cleaning} rows cleaned down to an empty string "
            f"and were kept, not dropped."
        )

    y_pred = predict_labels(texts, args.model_dir, args.batch_size)
    result = score(args.split, y_true, y_pred, empty_after_cleaning)

    print()
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(ID_TO_DIALECT))),
            target_names=list(ID_TO_DIALECT),
            digits=3,
            zero_division=0,
        )
    )
    print(confusion_as_markdown(result))
    print()

    write_reports(result, args.report_dir, plot=not args.no_plot)
    print(f"Reports written to {args.report_dir}")

    passed = result.macro_f1 >= args.min_macro_f1
    verdict = "PASS" if passed else "FAIL"
    print(
        f"{verdict}: macro F1 {result.macro_f1:.4f} "
        f"against a floor of {args.min_macro_f1:.4f}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
