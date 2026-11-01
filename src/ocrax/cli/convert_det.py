#!/usr/bin/env python3
"""Convert exported Paddle PIR PP-OCRv6 detection weights into an Orbax checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from ocrax.application.convert_det_pir import convert_detection

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARRAYS = REPO_ROOT / "artifacts/checkpoints/ppocrv6_medium_det_exported.npz"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/checkpoints/ppocrv6_medium_det_orbax"


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arrays",
        type=Path,
        default=DEFAULT_ARRAYS,
        help=f"Exported .npz file with Paddle parameter arrays (default: {DEFAULT_ARRAYS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination directory for the converted Orbax checkpoint (default: {DEFAULT_OUTPUT})",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _make_parser().parse_args(argv)
    manifest = convert_detection(args.arrays.resolve(), args.output.resolve())
    print(
        f"Converted {manifest['parameter_count']} parameter tensors to "
        f"Orbax checkpoint at {manifest['output_dir']}"
    )


if __name__ == "__main__":
    main()
