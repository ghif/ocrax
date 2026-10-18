"""RxHandBD manifest loading, generation, and validation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

def load_labels(dataset_root: Path, split: str) -> list[dict[str, str]]:
    if split != "train":
        raise ValueError("Only Train_Label.csv may be used to prepare train/validation manifests")
    csv_path = dataset_root / "Train_Label.csv"
    image_dir = dataset_root / "Train_Set"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"Images", "Text"}.issubset(rows[0]):
        raise ValueError(f"Expected Images and Text columns in {csv_path}")
    seen: set[str] = set()
    examples = []
    for row in rows:
        image_name = row["Images"].strip()
        text = row["Text"]
        if not image_name or not text:
            raise ValueError(f"Empty image name or transcript in {csv_path}: {row!r}")
        if image_name in seen:
            raise ValueError(f"Duplicate training image {image_name!r}")
        seen.add(image_name)
        image_path = image_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        examples.append({"image": str(image_path.resolve()), "text": text})
    return examples


def split_examples(
    examples: list[dict[str, str]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.05 <= validation_fraction <= 0.3:
        raise ValueError("validation_fraction must be between 0.05 and 0.30")
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    n_val = max(1, round(len(examples) * validation_fraction))
    val_indices = set(order[:n_val])
    train = [e for i, e in enumerate(examples) if i not in val_indices]
    validation = [e for i, e in enumerate(examples) if i in val_indices]
    assert not ({e["image"] for e in train} & {e["image"] for e in validation})
    return train, validation


def write_manifest(path: Path, examples: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in examples:
            encoded = json.dumps(item, ensure_ascii=False, sort_keys=True)
            f.write(encoded + "\n")


def manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_colab_manifests(manifest_dir: Path) -> None:
    """Validate the train/validation split contract before a Colab run."""
    metadata_path = manifest_dir / "split_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing split metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Split metadata must be a JSON object: {metadata_path}")
    if metadata.get("dataset") != "RxHandBD" or metadata.get("source") != "Train_Label.csv only":
        raise ValueError("Colab training manifests must be generated from RxHandBD Train_Label.csv only")
    if metadata.get("test_labels_read") is not False:
        raise ValueError("Split metadata must confirm that test labels were not read")

    split_paths = {}
    split_basenames = {}
    for split in ("train", "validation"):
        path = manifest_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} manifest: {path}")
        expected_hash = metadata.get(f"{split}_sha256")
        if not isinstance(expected_hash, str) or manifest_hash(path) != expected_hash:
            raise ValueError(f"{split} manifest does not match {metadata_path}")
        rows = load_manifest(path)
        expected_count = metadata.get(f"{split}_count")
        if type(expected_count) is not int or len(rows) != expected_count:
            raise ValueError(f"{split} manifest row count does not match {metadata_path}")
        images = set()
        basenames = set()
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"{split} manifest row {index} must be a JSON object")
            image = row.get("image")
            text = row.get("text")
            if not isinstance(image, str) or not image or not isinstance(text, str) or not text:
                raise ValueError(f"{split} manifest row {index} needs non-empty image and text fields")
            image_path = Path(image)
            if not image_path.name or image_path.parent.name != "Train_Set":
                raise ValueError(f"{split} manifest row {index} must reference an image in Train_Set")
            if image in images or image_path.name in basenames:
                raise ValueError(f"{split} manifest contains duplicate image {image!r}")
            images.add(image)
            basenames.add(image_path.name)
        split_paths[split] = images
        split_basenames[split] = basenames

    if split_paths["train"] & split_paths["validation"]:
        raise ValueError("Train and validation manifests contain overlapping image paths")
    if split_basenames["train"] & split_basenames["validation"]:
        raise ValueError("Train and validation manifests contain overlapping image basenames")


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    if not examples:
        raise ValueError(f"Manifest is empty: {path}")
    return examples


def validate_label_vocabulary(
    examples: list[dict[str, str]], valid_characters: set[str], *, split: str
) -> None:
    """Fail before training if a manifest has empty or unsupported labels."""
    for index, example in enumerate(examples, start=1):
        text = example.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{split} manifest row {index} has an empty or invalid text label")
        unsupported = next((character for character in text if character not in valid_characters), None)
        if unsupported is not None:
            image = example.get("image", "<unknown image>")
            raise ValueError(
                f"{split} label for {image!r} contains out-of-dictionary character {unsupported!r}"
            )
