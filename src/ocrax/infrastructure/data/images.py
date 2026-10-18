"""Image preprocessing compatible with PP-OCR recognition inference."""

from __future__ import annotations

import numpy as np
from PIL import Image

def resize_rec_image(
    image: Image.Image | np.ndarray,
    image_height: int = 48,
    image_width: int = 320,
) -> tuple[np.ndarray, int]:
    """PP-OCR RecResizeImg behavior: BGR, fit-height, right-pad, [-1, 1], CHW."""
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
        image = rgb[:, :, ::-1]  # PaddleOCR DecodeImage uses BGR.
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got {image.shape}")
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image dimensions {image.shape}")
    resized_w = min(image_width, max(1, int(np.ceil(image_height * w / h))))
    resized = Image.fromarray(image.astype(np.uint8)).resize(
        (resized_w, image_height), Image.Resampling.BILINEAR
    )
    resized = np.asarray(resized, dtype=np.float32)
    canvas = np.zeros((image_height, image_width, 3), dtype=np.float32)
    canvas[:, :resized_w] = resized
    canvas = canvas / 255.0
    canvas = (canvas - 0.5) / 0.5
    return np.transpose(canvas, (2, 0, 1)), resized_w
