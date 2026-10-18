"""Check Paddle and JAX inference parity."""

from __future__ import annotations

import argparse
from pathlib import Path

from ocrax.application.parity import compare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    compare(args.model_dir, args.checkpoint, args.images, atol=args.atol, rtol=args.rtol, report_path=args.report)


if __name__ == "__main__":
    main()
