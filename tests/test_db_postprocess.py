import numpy as np
from ocrax.application.db_postprocess import (
    box_score_fast,
    extract_boxes_from_prob_map,
    get_rotate_crop_image,
    order_points_clockwise,
    unclip,
)


def test_order_points_clockwise():
    # Disordered rectangle corners
    pts = np.array([[10, 30], [50, 10], [10, 10], [50, 30]], dtype=np.float32)
    ordered = order_points_clockwise(pts)
    np.testing.assert_allclose(ordered[0], [10, 10])  # TL
    np.testing.assert_allclose(ordered[1], [50, 10])  # TR
    np.testing.assert_allclose(ordered[2], [50, 30])  # BR
    np.testing.assert_allclose(ordered[3], [10, 30])  # BL


def test_unclip():
    box = np.array([[20, 20], [60, 20], [60, 40], [20, 40]], dtype=np.float32)
    expanded = unclip(box, unclip_ratio=1.5)
    assert expanded is not None
    assert len(expanded) > 0


def test_box_score_fast():
    prob_map = np.ones((100, 100), dtype=np.float32) * 0.8
    box = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32)
    score = box_score_fast(prob_map, box)
    assert abs(score - 0.8) < 1e-4


def test_extract_boxes_from_prob_map():
    prob_map = np.zeros((100, 100), dtype=np.float32)
    # Text region with high probability
    prob_map[20:50, 20:80] = 0.95

    boxes, scores = extract_boxes_from_prob_map(
        prob_map, orig_shape=(100, 100), thresh=0.2, box_thresh=0.5
    )
    assert len(boxes) >= 1
    assert scores[0] > 0.5


def test_get_rotate_crop_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:50, 20:80] = 255
    box = np.array([[20, 20], [80, 20], [80, 50], [20, 50]], dtype=np.float32)
    crop = get_rotate_crop_image(img, box)
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0
