import numpy as np
from ocrax.application.reading_order import sort_reading_order


def test_reading_order_two_lines():
    # Line 1: two boxes at y ~ 20 (x=10..30 and x=50..70)
    box_l1_w1 = np.array([[10, 15], [30, 15], [30, 25], [10, 25]], dtype=np.float32)
    box_l1_w2 = np.array([[50, 16], [70, 16], [70, 26], [50, 26]], dtype=np.float32)

    # Line 2: two boxes at y ~ 50 (x=10..30 and x=50..70)
    box_l2_w1 = np.array([[10, 48], [30, 48], [30, 58], [10, 58]], dtype=np.float32)
    box_l2_w2 = np.array([[50, 49], [70, 49], [70, 59], [50, 59]], dtype=np.float32)

    # Input out of order: line2 word2, line1 word2, line2 word1, line1 word1
    shuffled = [box_l2_w2, box_l1_w2, box_l2_w1, box_l1_w1]
    sorted_boxes, _, indices = sort_reading_order(shuffled)

    assert len(sorted_boxes) == 4
    # Expected order: box_l1_w1 (idx 3), box_l1_w2 (idx 1), box_l2_w1 (idx 2), box_l2_w2 (idx 0)
    assert indices == [3, 1, 2, 0]


def test_reading_order_empty():
    boxes, scores, indices = sort_reading_order([])
    assert boxes == []
    assert indices == []
