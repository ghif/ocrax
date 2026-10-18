"""Train PP-OCRv6 medium with CTC on RxHandBD manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from ocrax.application.training import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/ppocrv6_medium_rec_orbax"))
    parser.add_argument("--train-manifest", type=Path, default=Path("data/manifests/train.jsonl"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("data/manifests/validation.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/ppocrv6_medium_rec_rxhandbd_finetuned"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20261017)
    return parser


def main() -> None:
    run_training(build_parser().parse_args())


if __name__ == "__main__":
    main()
