"""Orbax checkpoint persistence for the PP-OCRv6 NNX model."""

from __future__ import annotations

from pathlib import Path

from flax import nnx
import orbax.checkpoint as ocp

from ocrax.infrastructure.models.ppocrv6 import PP_OCRv6MediumRec
from ocrax.infrastructure.models.ppocrv6_det import PP_OCRv6MediumDet


def restore_nnx(checkpoint: Path) -> PP_OCRv6MediumRec:
    """Restore an Orbax NNX state tree into the current model graph."""
    model = PP_OCRv6MediumRec()
    _, target = nnx.split(model)
    restored = ocp.StandardCheckpointer().restore(checkpoint.resolve(), target=target)
    nnx.update(model, restored)
    return model


def restore_det_nnx(checkpoint: Path) -> PP_OCRv6MediumDet:
    """Restore an Orbax NNX state tree for the detection model."""
    model = PP_OCRv6MediumDet()
    _, target = nnx.split(model)
    restored = ocp.StandardCheckpointer().restore(checkpoint.resolve(), target=target)
    nnx.update(model, restored)
    return model


def save_state(path: Path, state, force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path, state, force=force)
    checkpointer.wait_until_finished()
