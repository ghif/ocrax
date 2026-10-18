"""Add requested fetch variables to a Paddle PIR graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from ocrax.infrastructure.paddle.probe import add_fetches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("probes", nargs="+", type=int)
    args = parser.parse_args()
    add_fetches(args.model_json, args.output_json, [(probe, f"probe_{probe}") for probe in args.probes])


if __name__ == "__main__":
    main()
