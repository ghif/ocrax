from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ocrax.infrastructure.data.manifests import validate_colab_manifests, validate_label_vocabulary


class LabelVocabularyTests(unittest.TestCase):
    def test_accepts_labels_with_supported_characters(self) -> None:
        validate_label_vocabulary(
            [{"image": "P1.jpg", "text": "Rx 123"}],
            set("Rx 123"),
            split="train",
        )

    def test_reports_unsupported_characters_before_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "P2.jpg.*'Ω'"):
            validate_label_vocabulary(
                [{"image": "P1.jpg", "text": "valid"}, {"image": "P2.jpg", "text": "Rx Ω"}],
                set("validRx "),
                split="train",
            )

    def test_rejects_empty_or_non_string_labels(self) -> None:
        for example in ({"image": "P1.jpg", "text": ""}, {"image": "P2.jpg", "text": None}):
            with self.subTest(example=example), self.assertRaisesRegex(ValueError, "row 1"):
                validate_label_vocabulary([example], set("Rx "), split="validation")


class ColabManifestTests(unittest.TestCase):
    def _write_manifests(self, directory: Path, *, test_labels_read: bool = False) -> None:
        train_rows = [{"image": "/data/Train_Set/A.jpg", "text": "A"}]
        validation_rows = [{"image": "/data/Train_Set/B.jpg", "text": "B"}]
        hashes = {}
        for split, rows in (("train", train_rows), ("validation", validation_rows)):
            manifest = directory / f"{split}.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
            )
            hashes[f"{split}_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            hashes[f"{split}_count"] = len(rows)
        metadata = {
            "dataset": "RxHandBD",
            "source": "Train_Label.csv only",
            "test_labels_read": test_labels_read,
            **hashes,
        }
        (directory / "split_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def test_validates_manifest_hashes_counts_and_test_split_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest_dir = Path(temp)
            self._write_manifests(manifest_dir)
            validate_colab_manifests(manifest_dir)

    def test_rejects_test_labels_or_stale_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest_dir = Path(temp)
            self._write_manifests(manifest_dir, test_labels_read=True)
            with self.assertRaisesRegex(ValueError, "test labels"):
                validate_colab_manifests(manifest_dir)
            self._write_manifests(manifest_dir)
            (manifest_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_colab_manifests(manifest_dir)


if __name__ == "__main__":
    unittest.main()
