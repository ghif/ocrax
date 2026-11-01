"""Reading order sorting for detected document text boxes."""

from __future__ import annotations

import numpy as np


def sort_reading_order(
    boxes: list[np.ndarray],
    scores: list[float] | None = None,
    line_threshold_ratio: float = 0.5,
) -> tuple[list[np.ndarray], list[float] | None, list[int]]:
    """Sort text boxes in top-to-bottom, left-to-right reading order.
    
    Args:
        boxes: List of 4x2 polygon arrays.
        scores: Optional list of confidence scores matching boxes.
        line_threshold_ratio: Max vertical distance relative to box height to group into same line.
        
    Returns:
        Tuple of (sorted_boxes, sorted_scores, original_indices).
    """
    if not boxes:
        return [], scores, []

    if scores is None:
        scores = [1.0] * len(boxes)

    items = []
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        ymin = float(box[:, 1].min())
        ymax = float(box[:, 1].max())
        xmin = float(box[:, 0].min())
        xmax = float(box[:, 0].max())
        cy = (ymin + ymax) / 2.0
        cx = (xmin + xmax) / 2.0
        h = max(1.0, ymax - ymin)
        items.append({
            "idx": idx,
            "box": box,
            "score": score,
            "cx": cx,
            "cy": cy,
            "ymin": ymin,
            "ymax": ymax,
            "xmin": xmin,
            "h": h,
        })

    # Sort initially by ymin
    items.sort(key=lambda x: x["ymin"])

    # Group into lines
    lines: list[list[dict]] = []
    for item in items:
        matched_line = None
        for line in lines:
            # Average height and cy of the line
            line_cy = sum(it["cy"] for it in line) / len(line)
            line_h = sum(it["h"] for it in line) / len(line)
            threshold = line_h * line_threshold_ratio
            if abs(item["cy"] - line_cy) <= threshold:
                matched_line = line
                break
        if matched_line is not None:
            matched_line.append(item)
        else:
            lines.append([item])

    # Sort each line left-to-right (by xmin or cx)
    sorted_items = []
    for line in lines:
        line.sort(key=lambda x: x["xmin"])
        sorted_items.extend(line)

    sorted_boxes = [item["box"] for item in sorted_items]
    sorted_scores = [item["score"] for item in sorted_items] if scores is not None else None
    original_indices = [item["idx"] for item in sorted_items]

    return sorted_boxes, sorted_scores, original_indices
