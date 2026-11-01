"""Pure JAX End-to-End OCR Pipeline combining PP-OCRv6 detection and fine-tuned recognition."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flax import nnx

from ocrax.application.db_postprocess import (
    extract_boxes_from_prob_map,
    get_rotate_crop_image,
)
from ocrax.application.reading_order import sort_reading_order
from ocrax.cli.evaluate_jax import _decode_batch, _load_characters
from ocrax.infrastructure.checkpoints.orbax import restore_det_nnx, restore_nnx
from ocrax.infrastructure.data.images import resize_rec_image

DEFAULT_DET_CHECKPOINT = Path("artifacts/checkpoints/ppocrv6_medium_det_orbax")
DEFAULT_REC_CHECKPOINT = Path(
    "artifacts/checkpoints/ppocrv6_medium_rec_rxhandbd_converged"
)
DEFAULT_MODEL_DIR = Path(
    "/Users/mghifary/.paddlex/official_models/PP-OCRv6_medium_rec"
)


@dataclass
class OCRItem:
    index: int
    box: np.ndarray
    det_score: float
    text: str
    rec_score: float
    crop: Image.Image


@dataclass
class OCRDocumentResult:
    items: list[OCRItem]
    full_text: str
    t_det_ms: float
    t_rec_ms: float
    t_total_ms: float
    original_size: tuple[int, int]


def resize_image_det(
    img: np.ndarray, max_side_len: int = 960
) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize image so max side <= max_side_len and dimensions are multiples of 32."""
    h, w = img.shape[:2]
    ratio = 1.0
    if max(h, w) > max_side_len:
        if h > w:
            ratio = float(max_side_len) / h
        else:
            ratio = float(max_side_len) / w
    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)

    resized_img = cv2.resize(img, (resize_w, resize_h))
    return resized_img, (h, w)


def normalize_det_image(img: np.ndarray) -> np.ndarray:
    """Normalize image with ImageNet mean and std for detection (returns NCHW float32)."""
    # img is RGB uint8
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    norm = (img.astype(np.float32) / 255.0 - mean) / std
    return np.transpose(norm, (2, 0, 1))[np.newaxis, ...]  # Shape (1, 3, H, W)


class InteractiveOCRPipeline:
    """Pure JAX End-to-End Pipeline for Document Text Detection and Recognition."""

    def __init__(
        self,
        det_checkpoint: Path = DEFAULT_DET_CHECKPOINT,
        rec_checkpoint: Path = DEFAULT_REC_CHECKPOINT,
        model_dir: Path = DEFAULT_MODEL_DIR,
    ):
        print(f"Loading JAX detection model from {det_checkpoint}...")
        self.det_model = restore_det_nnx(det_checkpoint.resolve())
        print(f"Loading JAX recognition model from {rec_checkpoint}...")
        self.rec_model = restore_nnx(rec_checkpoint.resolve())
        self.characters = _load_characters(model_dir.resolve())

        # Compile JAX inference functions
        self._det_fn = nnx.jit(lambda m, x: m(x))
        self._rec_fn = nnx.jit(lambda m, x: m(x, training=False))

        # Warmup JIT compilation
        print("Warmup JAX JIT compilation on CPU...")
        dummy_det = jnp.zeros((1, 3, 64, 64), dtype=jnp.float32)
        self._det_fn(self.det_model, dummy_det).block_until_ready()
        dummy_rec = jnp.zeros((1, 3, 48, 320), dtype=jnp.float32)
        self._rec_fn(self.rec_model, dummy_rec).block_until_ready()
        print("Pipeline is ready!")

    def process_image(
        self,
        image_input: Image.Image | np.ndarray,
        box_thresh: float = 0.45,
        det_thresh: float = 0.2,
        unclip_ratio: float = 1.4,
        max_side_len: int = 960,
        batch_size: int = 16,
    ) -> OCRDocumentResult:
        """Run pure JAX end-to-end OCR on an input document image."""
        t_start = time.perf_counter()

        if isinstance(image_input, Image.Image):
            rgb_np = np.array(image_input.convert("RGB"))
        else:
            rgb_np = image_input
            if rgb_np.ndim == 2:
                rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_GRAY2RGB)

        orig_h, orig_w = rgb_np.shape[:2]

        # 1. Detection preprocessing
        resized_img, orig_shape = resize_image_det(rgb_np, max_side_len=max_side_len)
        norm_det = normalize_det_image(resized_img)

        # 2. JAX Detection inference
        t0_det = time.perf_counter()
        prob_map_jax = self._det_fn(self.det_model, jnp.asarray(norm_det))
        prob_map_jax.block_until_ready()
        t_det_ms = (time.perf_counter() - t0_det) * 1000.0

        # Extract 2D probability map
        prob_map = np.array(prob_map_jax[0, 0])

        # 3. Postprocess detection
        boxes, det_scores = extract_boxes_from_prob_map(
            prob_map,
            orig_shape,
            thresh=det_thresh,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
        )

        # Fallback if no boxes detected (e.g., pre-cropped word)
        if not boxes:
            h, w = orig_shape
            boxes = [np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)]
            det_scores = [1.0]

        # 4. Reading order sorting
        sorted_boxes, sorted_scores, _ = sort_reading_order(boxes, det_scores)

        # 5. Crop text regions
        crops: list[Image.Image] = []
        for box in sorted_boxes:
            crop_np = get_rotate_crop_image(rgb_np, box)
            crops.append(Image.fromarray(crop_np))

        # 6. JAX Recognition inference
        t0_rec = time.perf_counter()
        rec_texts: list[str] = []
        rec_confs: list[float] = []

        for start in range(0, len(crops), batch_size):
            batch_crops = crops[start : start + batch_size]
            tensors = [resize_rec_image(c)[0] for c in batch_crops]
            batch_arr = np.stack(tensors).astype(np.float32)
            logits = self._rec_fn(self.rec_model, jnp.asarray(batch_arr))
            logits.block_until_ready()
            texts, confs = _decode_batch(logits, self.characters)
            rec_texts.extend(texts)
            rec_confs.extend(confs)

        t_rec_ms = (time.perf_counter() - t0_rec) * 1000.0
        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        items: list[OCRItem] = []
        for i, (box, d_score, text, r_score, crop) in enumerate(
            zip(sorted_boxes, sorted_scores or [], rec_texts, rec_confs, crops),
            start=1,
        ):
            items.append(
                OCRItem(
                    index=i,
                    box=box,
                    det_score=float(d_score),
                    text=text,
                    rec_score=float(r_score),
                    crop=crop,
                )
            )

        full_text = " ".join(item.text for item in items if item.text.strip())

        return OCRDocumentResult(
            items=items,
            full_text=full_text,
            t_det_ms=t_det_ms,
            t_rec_ms=t_rec_ms,
            t_total_ms=t_total_ms,
            original_size=(orig_w, orig_h),
        )

    @staticmethod
    def draw_overlay(
        image: Image.Image,
        items: list[OCRItem],
        alpha: float = 0.25,
        line_width: int = 2,
        show_badges: bool = True,
    ) -> Image.Image:
        """Create visualization overlay with segmentation polygons and badges."""
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_base = ImageDraw.Draw(base)

        colors = [
            (0, 229, 255),    # Vibrant Cyan
            (16, 185, 129),   # Emerald Green
            (245, 158, 11),   # Amber
            (168, 85, 247),   # Purple
            (236, 72, 153),   # Pink
            (59, 130, 246),   # Royal Blue
        ]

        for item in items:
            box = item.box
            poly_tuples = [(float(p[0]), float(p[1])) for p in box]
            color = colors[(item.index - 1) % len(colors)]
            fill_color = (*color, int(255 * alpha))
            border_color = (*color, 255)

            # Draw semi-transparent fill on overlay
            draw_overlay.polygon(poly_tuples, fill=fill_color)

            # Draw crisp border
            draw_base.line(poly_tuples + [poly_tuples[0]], fill=border_color, width=line_width)

            if show_badges:
                # Place numbered badge at top-left corner
                tl = poly_tuples[0]
                badge_x = int(tl[0])
                badge_y = max(0, int(tl[1]) - 18)
                label = f"{item.index}"
                badge_w = max(18, len(label) * 8 + 8)
                badge_h = 16
                draw_base.rounded_rectangle(
                    [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
                    radius=3,
                    fill=(15, 23, 42, 230),  # Dark navy
                    outline=border_color,
                    width=1,
                )
                draw_base.text(
                    (badge_x + 4, badge_y + 1),
                    label,
                    fill=(255, 255, 255, 255),
                )

        combined = Image.alpha_composite(base, overlay)
        return combined.convert("RGB")
