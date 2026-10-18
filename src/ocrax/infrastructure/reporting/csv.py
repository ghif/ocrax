"""CSV schema helpers shared by OCR evaluation commands."""

from __future__ import annotations

import csv
from typing import TextIO

PREDICTION_FIELDS = ("image", "ground_truth", "prediction", "confidence", "correct")


def prediction_writer(stream: TextIO) -> csv.DictWriter:
    writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
    writer.writeheader()
    return writer
