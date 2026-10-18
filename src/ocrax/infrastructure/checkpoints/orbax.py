"""Orbax checkpoint persistence for the PP-OCRv6 NNX model."""

from __future__ import annotations

from pathlib import Path

from flax import nnx
import orbax.checkpoint as ocp

from ocrax.infrastructure.models.ppocrv6 import PP_OCRv6MediumRec


def restore_nnx(checkpoint: Path) -> PP_OCRv6MediumRec:
    """Restore an Orbax NNX state tree into the current model graph."""
    model = PP_OCRv6MediumRec()
    _, target = nnx.split(model)
    restored = ocp.StandardCheckpointer().restore(checkpoint, target=target)
    nnx.update(model, restored)
    return model


def save_state(path: Path, state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path, state)
    checkpointer.wait_until_finished()
