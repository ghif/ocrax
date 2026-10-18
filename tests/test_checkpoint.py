from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from flax import nnx

from ocrax.infrastructure.checkpoints.orbax import restore_nnx, save_state
from ocrax.infrastructure.models.ppocrv6 import PP_OCRv6MediumRec


class OrbaxCheckpointTests(unittest.TestCase):
    def test_state_round_trip_restores_model_parameters(self) -> None:
        source = PP_OCRv6MediumRec()
        _, source_state = nnx.split(source)
        source_flat = dict(nnx.to_flat_state(source_state))
        first_path = next(iter(source_flat))
        expected = np.asarray(source_flat[first_path].get_value())

        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "checkpoint"
            save_state(checkpoint, source_state)
            restored = restore_nnx(checkpoint)

        _, restored_state = nnx.split(restored)
        restored_flat = dict(nnx.to_flat_state(restored_state))
        np.testing.assert_array_equal(np.asarray(restored_flat[first_path].get_value()), expected)


if __name__ == "__main__":
    unittest.main()
