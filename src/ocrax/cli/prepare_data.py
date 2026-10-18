"""Prepare train and validation manifests from RxHandBD training labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocrax.infrastructure.data.manifests import (
    load_labels, split_examples, write_manifest, manifest_hash,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    examples = load_labels(args.dataset_root.expanduser().resolve(), "train")
    train, validation = split_examples(examples, args.validation_fraction, args.seed)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_manifest(train_path, train)
    write_manifest(validation_path, validation)
    metadata = {
        "dataset": "RxHandBD", "source": "Train_Label.csv only",
        "seed": args.seed, "validation_fraction": args.validation_fraction,
        "split_unit": "sample; dataset provides no writer/source-group column",
        "train_count": len(train), "validation_count": len(validation),
        "train_sha256": manifest_hash(train_path),
        "validation_sha256": manifest_hash(validation_path),
        "test_labels_read": False,
    }
    (args.output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    chars = set("".join(e["text"] for e in examples))
    print(json.dumps({**metadata, "label_char_count": len(chars), "characters": "".join(sorted(chars))}, indent=2))


if __name__ == "__main__":
    main()
