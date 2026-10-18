#!/usr/bin/env python3
"""Evaluate the local JAX PP-OCRv6 medium recognizer on RxHandBD word crops.

Requires JAX, Flax NNX, Orbax Checkpoint, NumPy, Pillow, and PyYAML. The
default Orbax checkpoint is the converted inference checkpoint in
``artifacts/checkpoints/ppocrv6_medium_rec_orbax``. Run on CPU with:

    JAX_PLATFORMS=cpu conda run -n jax-cpu python \
        ocrax-evaluate-jax

Metrics and CSV fields match ``ocrax-evaluate``. Inference uses
the port's parity-checked PP-OCRv6 CTC path, without Paddle inference.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from ocrax.application.evaluation import score_predictions
from ocrax.domain.metrics import normalize_text
from ocrax.infrastructure.data.evaluation import load_ground_truth as _load_ground_truth
from ocrax.infrastructure.reporting.csv import prediction_writer


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = Path(
    "/Users/mghifary/Work/Code/AI/data/18478741/RxHandBD"
)
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts/checkpoints/ppocrv6_medium_rec_orbax"
DEFAULT_MODEL_DIR = Path(
    "/Users/mghifary/.paddlex/official_models/PP-OCRv6_medium_rec"
)
DEFAULT_OUTPUT = Path("outputs/rxhandbd_ppocrv6_jax_predictions.csv")


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"RxHandBD directory (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Converted Orbax checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"PP-OCRv6 inference directory used for its character dictionary (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32, matching the Paddle baseline)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Per-image prediction CSV (default: {DEFAULT_OUTPUT})",
    )
    return parser


def _load_characters(model_dir: Path) -> list[str]:
    config_path = model_dir / "inference.yml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"PP-OCRv6 character dictionary config not found: {config_path}"
        )
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load inference.yml") from exc
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        characters = list(config["PostProcess"]["character_dict"])
    except (TypeError, KeyError) as exc:
        raise ValueError(
            f"Could not read PostProcess.character_dict from {config_path}"
        ) from exc
    # Paddle's CTCLabelDecode adds a space token, then prepends the blank ID.
    if config["PostProcess"].get("use_space_char", True):
        characters.append(" ")
    return characters


def _restore_model(checkpoint: Path):
    try:
        from flax import nnx
        from ocrax.infrastructure.checkpoints.orbax import restore_nnx
    except ImportError as exc:
        raise RuntimeError(
            "JAX, Flax NNX, and Orbax Checkpoint are required. Use an environment "
            "with the dependencies listed in README.md."
        ) from exc
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"JAX Orbax checkpoint not found: {checkpoint}")
    return restore_nnx(checkpoint.resolve()), nnx


def _decode_batch(probabilities, characters: list[str]) -> tuple[list[str], list[float]]:
    import numpy as np

    probs = np.asarray(probabilities)
    ids = probs.argmax(axis=-1)
    scores = probs.max(axis=-1)
    texts: list[str] = []
    confidences: list[float] = []
    for token_row, score_row in zip(ids, scores, strict=True):
        previous = None
        chars: list[str] = []
        char_scores: list[float] = []
        for token, score in zip(token_row.tolist(), score_row.tolist(), strict=True):
            if token != previous and token != 0:
                char_index = token - 1
                if char_index >= len(characters):
                    raise RuntimeError(
                        f"CTC token {token} exceeds character dictionary size "
                        f"{len(characters)}"
                    )
                chars.append(characters[char_index])
                char_scores.append(float(score))
            previous = token
        texts.append("".join(chars))
        confidences.append(sum(char_scores) / len(char_scores) if char_scores else 0.0)
    return texts, confidences


def evaluate(args: argparse.Namespace) -> dict[str, float | int]:
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

    checkpoint = args.checkpoint.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    characters = _load_characters(model_dir)
    model, nnx = _restore_model(checkpoint)
    from jax import numpy as jnp
    import numpy as np
    from PIL import Image

    from ocrax.infrastructure.data.images import resize_rec_image

    infer = nnx.jit(lambda model_arg, images: model_arg(images, training=False))
    batch_sizes = {min(args.batch_size, len(rows))}
    remainder = len(rows) % args.batch_size
    if remainder:
        batch_sizes.add(remainder)
    print(
        f"Loaded {len(rows)} labeled images and the local JAX checkpoint "
        f"{checkpoint}; preparing {len(characters)} CTC characters",
        flush=True,
    )
    print("Compiling JAX CPU inference shape(s)...", flush=True)
    for size in sorted(batch_sizes):
        warmup = jnp.zeros((size, 3, 48, 320), dtype=jnp.float32)
        infer(model, warmup).block_until_ready()

    predictions: list[dict[str, Any]] = []
    inference_seconds = 0.0
    last_progress = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as csv_file:
        writer = prediction_writer(csv_file)
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            inference_start = time.perf_counter()
            images = []
            for row in batch_rows:
                with Image.open(image_dir / row["image"]) as image:
                    images.append(resize_rec_image(image)[0])
            batch = np.stack(images).astype(np.float32, copy=False)
            probabilities = infer(model, jnp.asarray(batch))
            probabilities.block_until_ready()
            texts, confidences = _decode_batch(probabilities, characters)
            inference_seconds += time.perf_counter() - inference_start
            if len(texts) != len(batch_rows):
                raise RuntimeError(
                    f"JAX model returned {len(texts)} result(s) for "
                    f"{len(batch_rows)} input image(s) in batch starting at row {start + 1}"
                )
            for row, text, confidence in zip(
                batch_rows, texts, confidences, strict=True
            ):
                correct = text == row["ground_truth"]
                prediction = {
                    "image": row["image"],
                    "ground_truth": row["ground_truth"],
                    "prediction": text,
                    "confidence": confidence,
                    "raw_correct": correct,
                    "normalized_correct": normalize_text(text)
                    == normalize_text(row["ground_truth"]),
                }
                predictions.append(prediction)
                writer.writerow(
                    {
                        "image": row["image"],
                        "ground_truth": row["ground_truth"],
                        "prediction": text,
                        "confidence": confidence,
                        "correct": correct,
                    }
                )
            csv_file.flush()
            done = start + len(batch_rows)
            if done == len(batch_rows) or done - last_progress >= 100 or done == len(rows):
                print(f"Processed {done}/{len(rows)} images", flush=True)
                last_progress = done

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
    print(f"Inference time (prediction batches): {inference_seconds:.3f} s")
    print(f"Average inference latency: {average_latency_ms:.2f} ms/image")
    print(f"Inference throughput: {throughput:.2f} images/s")
    print(f"Predictions saved to: {args.output.resolve()}")
    return metrics


def main() -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    args = _make_parser().parse_args()
    try:
        evaluate(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
