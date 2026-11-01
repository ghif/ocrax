from pathlib import Path
from PIL import Image
import numpy as np

from ocrax.application.pipeline import (
    DEFAULT_DET_CHECKPOINT,
    DEFAULT_REC_CHECKPOINT,
    InteractiveOCRPipeline,
)


def test_pipeline_on_image():
    if not DEFAULT_DET_CHECKPOINT.is_dir() or not DEFAULT_REC_CHECKPOINT.is_dir():
        return

    pipeline = InteractiveOCRPipeline(
        det_checkpoint=DEFAULT_DET_CHECKPOINT,
        rec_checkpoint=DEFAULT_REC_CHECKPOINT,
    )

    # Create synthetic test image with a white box
    img = Image.new("RGB", (128, 64), color=(0, 0, 0))
    res = pipeline.process_image(img)

    assert res is not None
    assert len(res.items) >= 1
    assert res.t_det_ms > 0
    assert res.t_rec_ms > 0

    overlay = pipeline.draw_overlay(img, res.items)
    assert overlay.size == img.size
