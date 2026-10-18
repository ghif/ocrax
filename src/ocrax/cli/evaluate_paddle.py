#!/usr/bin/env python3
"""Evaluate PP-OCRv6 on RxHandBD's handwritten word crops.

Requires PaddleOCR 3.7 or newer. The first run may download the pretrained
``PP-OCRv6_medium_rec`` model.

Example::

    ocrax-evaluate \
        --dataset-root "/Users/mghifary/Work/Code/AI/data/18478741/RxHandBD"
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

from ocrax.application.evaluation import score_predictions
from ocrax.domain.metrics import normalize_text
from ocrax.infrastructure.data.evaluation import load_ground_truth as _load_ground_truth
from ocrax.infrastructure.reporting.csv import prediction_writer


DEFAULT_DATASET_ROOT = Path(
    "/Users/mghifary/Work/Code/AI/data/18478741/RxHandBD"
)
DEFAULT_OUTPUT = Path("outputs/rxhandbd_ppocrv6_predictions.csv")


def _result_payload(result: Any) -> dict[str, Any]:
    """Extract PaddleX's result payload across supported result wrappers."""
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if payload is None and isinstance(result, dict):
        payload = result
    if isinstance(payload, str):
        import json

        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError(f"Unrecognized PaddleOCR result type: {type(result).__name__}")
    nested = payload.get("res", payload)
    if not isinstance(nested, dict):
        raise ValueError("PaddleOCR result is missing its 'res' object")
    return nested


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"RxHandBD directory (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Inference batch size (default: 32)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Paddle device, such as cpu or gpu:0 (default: cpu)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Per-image prediction CSV (default: {DEFAULT_OUTPUT})",
    )
    return parser


def evaluate(args: argparse.Namespace) -> tuple[int, float, float, float, float]:
    dataset_root = args.dataset_root.expanduser().resolve()
    image_dir = dataset_root / "Test_Set"
    labels_path = dataset_root / "Test_Labels.csv"
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Test labels CSV not found: {labels_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {image_dir}")

    rows = _load_ground_truth(labels_path, image_dir)

    try:
        from paddleocr import TextRecognition
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR 3.7 or newer is required. Install it with: "
            "python -m pip install 'paddleocr[all]>=3.7.0'"
        ) from exc

    model = TextRecognition(model_name="PP-OCRv6_medium_rec", device=args.device)
    predictions: list[dict[str, Any]] = []
    inference_seconds = 0.0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        image_paths = [str(image_dir / row["image"]) for row in batch]
        inference_start = time.perf_counter()
        results = list(model.predict(input=image_paths, batch_size=args.batch_size))
        inference_seconds += time.perf_counter() - inference_start
        if len(results) != len(batch):
            raise RuntimeError(
                f"PaddleOCR returned {len(results)} result(s) for "
                f"{len(batch)} input image(s) in batch starting at row {start + 1}"
            )
        for row, result in zip(batch, results, strict=True):
            payload = _result_payload(result)
            if "rec_text" not in payload:
                raise RuntimeError(
                    f"PaddleOCR result for {row['image']} is missing 'rec_text'"
                )
            text = str(payload["rec_text"])
            confidence = payload.get("rec_score")
            predictions.append(
                {
                    "image": row["image"],
                    "ground_truth": row["ground_truth"],
                    "prediction": text,
                    "confidence": confidence,
                    "raw_correct": text == row["ground_truth"],
                    "normalized_correct": normalize_text(text)
                    == normalize_text(row["ground_truth"]),
                }
            )

    score = score_predictions(predictions)
    count = score["count"]
    raw_accuracy = score["raw_accuracy"]
    normalized_accuracy = score["normalized_accuracy"]
    cer = score["cer"]
    word_level_accuracy = score["word_level_accuracy"]
    average_latency_ms = inference_seconds * 1000 / count
    throughput = count / inference_seconds if inference_seconds else math.inf

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as csv_file:
        writer = prediction_writer(csv_file)
        for row in predictions:
            writer.writerow(
                {
                    "image": row["image"],
                    "ground_truth": row["ground_truth"],
                    "prediction": row["prediction"],
                    "confidence": row["confidence"],
                    "correct": row["raw_correct"],
                }
            )

    print(f"Evaluated images: {count}")
    print(f"Raw exact-match accuracy: {raw_accuracy:.4%}")
    print(f"Normalized exact-match accuracy: {normalized_accuracy:.4%}")
    print(f"Character error rate (CER): {cer:.4%}")
    print(f"Word-level accuracy (1 - WER): {word_level_accuracy:.4%}")
    print(f"Inference time (prediction calls): {inference_seconds:.3f} s")
    print(f"Average inference latency: {average_latency_ms:.2f} ms/image")
    print(f"Inference throughput: {throughput:.2f} images/s")
    print(f"Predictions saved to: {args.output.resolve()}")
    return count, raw_accuracy, normalized_accuracy, cer, word_level_accuracy


def main() -> int:
    parser = _make_parser()
    args = parser.parse_args()
    try:
        evaluate(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
