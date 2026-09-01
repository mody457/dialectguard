"""Model loading and single-text inference.

The classifier is loaded once at application startup and held in memory for
the process lifetime, because loading MARBERTv2 takes seconds and would
otherwise be paid on every request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from fastapi import status
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import ID_TO_DIALECT, MAX_SEQUENCE_LENGTH
from app.errors import APIError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    """A predicted dialect and the softmax probability behind it."""

    dialect: str
    confidence: float


class DialectClassifier:
    """Wraps the fine-tuned MARBERTv2 checkpoint for single-text inference."""

    def __init__(self, tokenizer: AutoTokenizer, model: torch.nn.Module, model_dir: Path):
        self._tokenizer = tokenizer
        self._model = model
        self.model_dir = model_dir

    @classmethod
    def load(cls, model_dir: Path) -> "DialectClassifier":
        """Load tokenizer and weights from a local directory.

        The label count is checked against ID_TO_DIALECT here rather than at
        prediction time, so a mismatched checkpoint fails at startup instead of
        producing silently wrong country codes under load.
        """
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

        expected_labels = len(ID_TO_DIALECT)
        if model.config.num_labels != expected_labels:
            raise ValueError(
                f"Checkpoint has {model.config.num_labels} labels, "
                f"expected {expected_labels} ({', '.join(ID_TO_DIALECT)})."
            )

        # The shipped config.json carries placeholder LABEL_0..LABEL_5 names.
        # Overwriting them in memory keeps the generated OpenAPI docs and any
        # direct use of the model object consistent with the API responses.
        model.config.id2label = dict(enumerate(ID_TO_DIALECT))
        model.config.label2id = {name: i for i, name in enumerate(ID_TO_DIALECT)}

        model.eval()
        logger.info(
            "model_loaded",
            extra={"context": {"model_dir": str(model_dir), "num_labels": expected_labels}},
        )
        return cls(tokenizer=tokenizer, model=model, model_dir=model_dir)

    def predict(self, cleaned_text: str) -> Prediction:
        """Classify already-preprocessed text.

        The caller is responsible for having run the training-time cleaning
        pipeline first. Passing raw text here reintroduces the train/serve skew
        the preprocessing exists to prevent.

        Raises:
            APIError: if the text exceeds the model's token budget. It is
                rejected rather than truncated so the caller never receives a
                confident label for a silently clipped sentence.
        """
        encoding = self._tokenizer(cleaned_text, truncation=False, padding=False)
        token_count = len(encoding["input_ids"])
        if token_count > MAX_SEQUENCE_LENGTH:
            raise APIError(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "TEXT_TOO_MANY_TOKENS",
                f"Field 'text' is {token_count} tokens after cleaning, "
                f"the model accepts at most {MAX_SEQUENCE_LENGTH}.",
            )

        inputs = {key: torch.tensor([value]) for key, value in encoding.items()}
        with torch.inference_mode():
            logits = self._model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_index = int(torch.argmax(probabilities))
        return Prediction(
            dialect=ID_TO_DIALECT[predicted_index],
            confidence=float(probabilities[predicted_index]),
        )
