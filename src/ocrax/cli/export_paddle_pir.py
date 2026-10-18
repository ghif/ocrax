"""Export Paddle PIR model parameters to an NPZ file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocrax.infrastructure.paddle.export_paddle_pir import export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.model_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
