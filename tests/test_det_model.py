from pathlib import Path
import jax.numpy as jnp
import numpy as np

from ocrax.infrastructure.checkpoints.orbax import restore_det_nnx
from ocrax.infrastructure.models.ppocrv6_det import PP_OCRv6MediumDet

DET_CHECKPOINT = Path("artifacts/checkpoints/ppocrv6_medium_det_orbax")


def test_det_model_initialization():
    model = PP_OCRv6MediumDet()
    assert len(model.params) == 257


def test_det_checkpoint_restoration_and_forward():
    if not DET_CHECKPOINT.is_dir():
        return
    model = restore_det_nnx(DET_CHECKPOINT)
    assert len(model.params) == 257

    # Run dummy forward pass
    dummy = jnp.zeros((1, 3, 64, 64), dtype=jnp.float32)
    out = model(dummy)
    assert out.shape == (1, 1, 64, 64)
    # Sigmoid output in [0, 1]
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
