"""Differentiable Binarization (DB) postprocessing for text detection."""

from __future__ import annotations

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """Order 4 polygon points clockwise starting from top-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def box_score_fast(bitmap: np.ndarray, _box: np.ndarray) -> float:
    """Calculate mean score inside a bounding box."""
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return float(cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0])


def unclip(box: np.ndarray, unclip_ratio: float = 1.4) -> np.ndarray | None:
    """Expand polygon box using pyclipper offset."""
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / (poly.length + 1e-6)
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    if not expanded:
        return None
    return np.array(expanded[0])


def extract_boxes_from_prob_map(
    prob_map: np.ndarray,
    orig_shape: tuple[int, int],
    thresh: float = 0.2,
    box_thresh: float = 0.45,
    max_candidates: int = 1000,
    unclip_ratio: float = 1.4,
    min_size: int = 3,
) -> tuple[list[np.ndarray], list[float]]:
    """Extract bounding polygons and confidence scores from probability heatmap.
    
    Args:
        prob_map: 2D float array (H_map, W_map) with values in [0, 1].
        orig_shape: Tuple (H_orig, W_orig) of original image.
        thresh: Binarization threshold for probability map.
        box_thresh: Minimum mean probability threshold for a box to be accepted.
        max_candidates: Max number of candidate contours to consider.
        unclip_ratio: Ratio for expanding detected polygon to cover full text region.
        min_size: Minimum width or height of bounding box.
        
    Returns:
        Tuple of (list of 4x2 polygon arrays, list of confidence scores).
    """
    h_map, w_map = prob_map.shape
    h_orig, w_orig = orig_shape
    scale_x = w_orig / float(w_map)
    scale_y = h_orig / float(h_map)

    # 1. Binarize probability map
    bitmap = (prob_map > thresh).astype(np.uint8)

    # 2. Extract contours
    contours, _ = cv2.findContours(
        bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    num_contours = min(len(contours), max_candidates)
    boxes = []
    scores = []

    for i in range(num_contours):
        contour = contours[i]
        points, sside = get_mini_boxes(contour)
        if sside < min_size:
            continue

        score = box_score_fast(prob_map, points)
        if score < box_thresh:
            continue

        expanded = unclip(points, unclip_ratio=unclip_ratio)
        if expanded is None:
            continue

        box, sside = get_mini_boxes(expanded)
        if sside < min_size + 2:
            continue

        # Scale coordinates back to original image resolution
        box[:, 0] = np.clip(box[:, 0] * scale_x, 0, w_orig)
        box[:, 1] = np.clip(box[:, 1] * scale_y, 0, h_orig)

        boxes.append(order_points_clockwise(box))
        scores.append(score)

    return boxes, scores


def get_mini_boxes(contour: np.ndarray) -> tuple[np.ndarray, float]:
    """Calculate minimum bounding rectangle for a contour."""
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return np.array(box, dtype=np.float32), min(bounding_box[1])


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Crop and rectify text polygon using perspective transform."""
    points = order_points_clockwise(points.astype(np.float32))
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
    )
    if img_crop_width <= 0:
        img_crop_width = 1
    if img_crop_height <= 0:
        img_crop_height = 1

    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    m = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        m,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_h, dst_w = dst_img.shape[0:2]
    if dst_h * 1.0 / max(1, dst_w) >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img
