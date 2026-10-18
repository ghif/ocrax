from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from ocrax.infrastructure.data.images import resize_rec_image


class PreprocessingTests(unittest.TestCase):
    def test_resize_preserves_ppocr_shape_bgr_normalization_and_padding(self) -> None:
        image = Image.new("RGB", (1, 1), color=(255, 0, 0))
        tensor, resized_width = resize_rec_image(image)
        self.assertEqual(tensor.shape, (3, 48, 320))
        self.assertEqual(resized_width, 48)
        np.testing.assert_allclose(tensor[:, 0, 0], np.array([-1.0, -1.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(tensor[:, 0, 48], np.full(3, -1.0, dtype=np.float32))

    def test_rejects_non_three_channel_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "3-channel"):
            resize_rec_image(np.zeros((48, 320), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
