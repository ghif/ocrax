"""Convert exported PP-OCRv6 medium detection inference tensors into an Orbax NNX state."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from flax import nnx

from ocrax.infrastructure.models.ppocrv6_det import PP_OCRv6MediumDet
from ocrax.infrastructure.checkpoints.orbax import save_state


def convert_detection(arrays_path: Path, output_dir: Path) -> dict:
    """Convert exported numpy weights into an Orbax checkpoint."""
    arrays_file = np.load(arrays_path)
    arrays = {key: arrays_file[key] for key in arrays_file.files}

    model = PP_OCRv6MediumDet(initial_params=arrays)
    _, state = nnx.split(model)

    save_state(output_dir, state, force=True)

    manifest = {
        "source_arrays": str(arrays_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "parameter_count": len(arrays),
        "parameters": {name: list(value.shape) for name, value in arrays.items()},
    }
    (output_dir / "conversion.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
