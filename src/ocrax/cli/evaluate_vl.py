#!/usr/bin/env python3
"""Evaluate the cached PaddleOCR-VL-0.9B model on RxHandBD word crops.

Uses PaddleOCR's native Paddle backend and an explicit local model directory.
It fails before initializing the pipeline if the checkpoint is unavailable;
there is no model-download fallback.

Requires PaddleOCR 3.7+, PaddleX 3.7.2 with OCR extras, and PaddlePaddle
3.3.1. Install PaddleX's optional pipeline dependencies with
``python -m pip install 'paddlex[ocr]==3.7.2'`` in the evaluation environment.

Example::

    conda run -n ocr ocrax-evaluate-vl \
        --device cpu \
        --output outputs/rxhandbd_paddleocr_vl_0_9_predictions.csv
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
DEFAULT_MODEL_DIR = Path(
    "/Users/mghifary/.paddlex/official_models/PaddleOCR-VL-0.9B"
)
DEFAULT_OUTPUT = Path("outputs/rxhandbd_paddleocr_vl_0_9_predictions.csv")
REQUIRED_MODEL_FILES = ("model.safetensors", "config.json", "inference.yml")


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"RxHandBD directory (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Existing PaddleOCR-VL-0.9B checkpoint directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Paddle device (default: cpu)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Per-image prediction CSV (default: {DEFAULT_OUTPUT})",
    )
    return parser


def _check_local_checkpoint(model_dir: Path) -> Path:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Local PaddleOCR-VL-0.9B checkpoint directory not found: {model_dir}. "
            "No download will be attempted."
        )
    missing = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Local checkpoint at {model_dir} is incomplete; missing: "
            f"{', '.join(missing)}. No download will be attempted."
        )
    return model_dir


def _extract_recognized_text(result: Any, image_name: str) -> str:
    """Join text blocks from the PaddleOCR-VL result for a whole-crop OCR prompt."""
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if payload is None and isinstance(result, dict):
        payload = result
    if isinstance(payload, str):
        import json

        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Unrecognized PaddleOCR-VL result type for {image_name}: "
            f"{type(result).__name__}"
        )
    payload = payload.get("res", payload)
    if not isinstance(payload, dict):
        raise RuntimeError(f"PaddleOCR-VL result for {image_name} has no result object")
    blocks = payload.get("parsing_res_list")
    if not isinstance(blocks, list):
        raise RuntimeError(
            f"PaddleOCR-VL result for {image_name} has no 'parsing_res_list'"
        )
    contents = [
        str(block.get("block_content", ""))
        for block in blocks
        if isinstance(block, dict) and block.get("block_label") == "ocr"
    ]
    return " ".join(content.strip() for content in contents if content.strip()).strip()


def evaluate(args: argparse.Namespace) -> dict[str, float | int]:
    dataset_root = args.dataset_root.expanduser().resolve()
    image_dir = dataset_root / "Test_Set"
    labels_path = dataset_root / "Test_Labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Test labels CSV not found: {labels_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {image_dir}")

    rows = _load_ground_truth(labels_path, image_dir)
    model_dir = _check_local_checkpoint(args.model_dir)
    print(f"Found {len(rows)} labeled images; using local checkpoint {model_dir}", flush=True)

    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR 3.7 or newer is required. Install PaddlePaddle and "
            "PaddleOCR in the active environment."
        ) from exc

    # Supplying the existing checkpoint directory prevents PaddleX from resolving
    # the VLM model through its model-hosting/download mechanism. Disable layout
    # and document preprocessing because every input is already a word crop.
    print("Initializing PaddleOCR-VL pipeline on " + args.device + "...", flush=True)
    pipeline = PaddleOCRVL(
        pipeline_version="v1",
        vl_rec_model_name="PaddleOCR-VL-0.9B",
        vl_rec_model_dir=str(model_dir),
        vl_rec_backend="native",
        use_layout_detection=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
        device=args.device,
    )

    predictions: list[dict[str, Any]] = []
    inference_seconds = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_file = args.output.open("w", encoding="utf-8", newline="")
    writer = prediction_writer(csv_file)
    try:
        for index, row in enumerate(rows, start=1):
            image_path = image_dir / row["image"]
            inference_start = time.perf_counter()
            results = list(
                pipeline.predict(
                    input=str(image_path),
                    use_layout_detection=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    prompt_label="ocr",
                    # Word crops have short transcriptions; bound generation so
                    # an illegible crop cannot spend minutes decoding a long answer.
                    max_new_tokens=64,
                )
            )
            inference_seconds += time.perf_counter() - inference_start
            if len(results) != 1:
                raise RuntimeError(
                    f"PaddleOCR-VL returned {len(results)} result(s) for "
                    f"{row['image']}; expected exactly one"
                )
            prediction = _extract_recognized_text(results[0], row["image"])
            predictions.append(
                {
                    "image": row["image"],
                    "ground_truth": row["ground_truth"],
                    "prediction": prediction,
                    "raw_correct": prediction == row["ground_truth"],
                    "normalized_correct": normalize_text(prediction)
                    == normalize_text(row["ground_truth"]),
                }
            )
            writer.writerow(
                {
                    "image": row["image"],
                    "ground_truth": row["ground_truth"],
                    "prediction": prediction,
                    "confidence": "",
                    "correct": prediction == row["ground_truth"],
                }
            )
            csv_file.flush()
            if index == 1 or index % 25 == 0 or index == len(rows):
                print(f"Processed {index}/{len(rows)} images", flush=True)
    finally:
        csv_file.close()
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()

    count = len(predictions)
    if count != len(rows):
        raise RuntimeError(
            f"Only {count} predictions were produced for {len(rows)} test labels"
        )
    score = score_predictions(predictions)
    raw_accuracy = score["raw_accuracy"]
    normalized_accuracy = score["normalized_accuracy"]
    cer = score["cer"]
    wer = score["wer"]
    word_level_accuracy = score["word_level_accuracy"]
    average_latency_ms = inference_seconds * 1000 / count
    throughput = count / inference_seconds if inference_seconds else math.inf

    metrics: dict[str, float | int] = {
        "images": count,
        "raw_accuracy": raw_accuracy,
        "normalized_accuracy": normalized_accuracy,
        "cer": cer,
        "wer": wer,
        "word_level_accuracy": word_level_accuracy,
        "inference_seconds": inference_seconds,
        "average_latency_ms": average_latency_ms,
        "throughput_images_per_second": throughput,
    }
    print(f"Evaluated images: {count}")
    print(f"Raw exact-match accuracy: {raw_accuracy:.4%}")
    print(f"Normalized exact-match accuracy: {normalized_accuracy:.4%}")
    print(f"Character error rate (CER): {cer:.4%}")
    print(f"Word-level accuracy (1 - WER): {word_level_accuracy:.4%}")
    print(f"Word error rate (WER): {wer:.4%}")
    print(f"Inference time (prediction calls): {inference_seconds:.3f} s")
    print(f"Average inference latency: {average_latency_ms:.2f} ms/image")
    print(f"Inference throughput: {throughput:.2f} images/s")
    print(f"Predictions saved to: {args.output.resolve()}")
    return metrics


def main() -> int:
    args = _make_parser().parse_args()
    try:
        evaluate(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
