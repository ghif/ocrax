"""Convert Paddle PIR tensors to an Orbax NNX checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocrax.application.convert_paddle_pir import convert


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.arrays, args.output), indent=2))


if __name__ == "__main__":
    main()
