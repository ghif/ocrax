from __future__ import annotations

import unittest

from ocrax.domain.metrics import (
    character_error_rate,
    edit_distance,
    normalize_text,
    word_error_rate,
)


class MetricTests(unittest.TestCase):
    def test_normalization_and_edit_distance(self) -> None:
        self.assertEqual(normalize_text("  Hello\tWORLD  "), "hello world")
        self.assertEqual(edit_distance("kitten", "sitting"), 3)

    def test_corpus_error_rates_aggregate_counts(self) -> None:
        self.assertAlmostEqual(character_error_rate(["cat", "dog"], ["cut", "dog"]), 1 / 6)
        self.assertAlmostEqual(word_error_rate(["a b", "c"], ["a x", "c"]), 1 / 3)


if __name__ == "__main__":
    unittest.main()
