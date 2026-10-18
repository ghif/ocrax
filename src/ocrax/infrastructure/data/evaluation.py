"""Ground-truth loading helpers for OCR evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

def load_ground_truth(labels_path: Path, image_dir: Path) -> list[dict[str, str]]:
    with labels_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"Images", "Text"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"{labels_path} must contain columns: Images, Text")
        rows = [{"image": row["Images"], "ground_truth": row["Text"]} for row in reader]
    if not rows:
        raise ValueError(f"No labeled examples found in {labels_path}")
    missing = [row["image"] for row in rows if not (image_dir / row["image"]).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise FileNotFoundError(f"{len(missing)} test image(s) listed in {labels_path} are missing from {image_dir}: {preview}{suffix}")
    return rows
