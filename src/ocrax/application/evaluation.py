"""Shared evaluation workflow for OCR text predictions."""

from __future__ import annotations

from typing import Sequence

from ocrax.domain.metrics import character_error_rate, normalize_text, word_error_rate


def score_predictions(predictions: Sequence[dict[str, object]]) -> dict[str, float | int]:
    """Compute the shared corpus metrics from per-image predictions."""
    references = [str(row["ground_truth"]) for row in predictions]
    hypotheses = [str(row["prediction"]) for row in predictions]
    count = len(predictions)
    raw_accuracy = sum(row["raw_correct"] for row in predictions) / count
    normalized_accuracy = sum(
        normalize_text(str(row["prediction"])) == normalize_text(str(row["ground_truth"]))
        for row in predictions
    ) / count
    cer = character_error_rate(references, hypotheses)
    wer = word_error_rate(references, hypotheses)
    return {
        "count": count,
        "raw_accuracy": raw_accuracy,
        "normalized_accuracy": normalized_accuracy,
        "cer": cer,
        "wer": wer,
        "word_level_accuracy": 1.0 - wer,
    }
