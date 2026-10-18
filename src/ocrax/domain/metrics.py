"""OCR text normalization and corpus metrics."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

def normalize_text(text: str) -> str:
    """Normalize case and collapse whitespace for tolerant comparisons."""
    return " ".join(text.casefold().split())


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    """Return Levenshtein edit distance using O(min(n, m)) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    """Compute corpus CER after case and whitespace normalization."""
    total_distance = 0
    total_reference_chars = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
        total_distance += edit_distance(reference, hypothesis)
        total_reference_chars += len(reference)
    if total_reference_chars == 0:
        return 0.0 if total_distance == 0 else math.inf
    return total_distance / total_reference_chars


def word_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    """Compute corpus WER over normalized whitespace-delimited words."""
    total_distance = 0
    total_reference_words = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        reference_words = normalize_text(reference).split()
        hypothesis_words = normalize_text(hypothesis).split()
        total_distance += edit_distance(reference_words, hypothesis_words)
        total_reference_words += len(reference_words)
    if total_reference_words == 0:
        return 0.0 if total_distance == 0 else math.inf
    return total_distance / total_reference_words
