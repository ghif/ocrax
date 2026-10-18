"""Continue fine-tuning from a stored Orbax checkpoint, locally or on Colab."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("local", "colab"))
    parser.add_argument("--epochs", type=int, default=3, help="New epochs to run from the stored weights")
    parser.add_argument("--gpu", choices=("T4", "L4", "G4", "A100", "H100"), default="T4", help="Colab GPU")
    parser.add_argument("--model-dir", type=Path, help="Local Paddle model directory containing inference.yml (required in local mode)")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/ppocrv6_medium_rec_rxhandbd_finetuned"), help="Local Orbax checkpoint for local mode")
    parser.add_argument("--train-manifest", type=Path, default=Path("data/manifests/train.jsonl"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("data/manifests/validation.jsonl"))
    parser.add_argument("--output", type=Path, help="New local output directory; must not already exist")
    parser.add_argument("--model-gcs", default="gs://handnotes-ocr/model-dir")
    parser.add_argument("--checkpoint-gcs", help="GCS Orbax checkpoint prefix for Colab mode")
    parser.add_argument("--dataset-gcs", default="gs://all-datasets/RxHandBD")
    parser.add_argument(
        "--output-gcs",
        help="Optional; Colab runs require MODEL_GCS/finetuned-runs and default there.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be a positive integer")
    if args.mode == "local" and args.model_dir is None:
        parser.error("local mode requires --model-dir")
    if args.mode == "colab" and not args.checkpoint_gcs:
        parser.error("colab mode requires --checkpoint-gcs pointing to the stored Orbax checkpoint prefix")
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "local":
        output = args.output or Path("artifacts/checkpoints/resumed_run")
        command = [
            sys.executable, "-m", "ocrax.cli.train",
            "--model-dir", str(args.model_dir),
            "--checkpoint", str(args.checkpoint),
            "--train-manifest", str(args.train_manifest),
            "--validation-manifest", str(args.validation_manifest),
            "--output", str(output),
            "--epochs", str(args.epochs),
            "--batch-size", "8", "--learning-rate", "1e-5", "--seed", "20261017",
        ]
    else:
        command = [
            sys.executable, "-m", "ocrax.cli.colab",
            "--model-gcs", args.model_gcs,
            "--checkpoint-gcs", args.checkpoint_gcs,
            "--dataset-gcs", args.dataset_gcs,
            "--gpu", args.gpu,
            "--epochs", str(args.epochs),
        ]
        if args.output_gcs:
            command.extend(["--output-gcs", args.output_gcs])
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
