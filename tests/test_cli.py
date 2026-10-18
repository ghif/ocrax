from __future__ import annotations

import unittest

from ocrax.cli.train import build_parser


class CliDefaultsTests(unittest.TestCase):
    def test_training_defaults_follow_relocated_artifact_paths(self) -> None:
        args = build_parser().parse_args(["--model-dir", "/models/ppocrv6"])
        self.assertEqual(str(args.checkpoint), "artifacts/checkpoints/ppocrv6_medium_rec_orbax")
        self.assertEqual(str(args.train_manifest), "data/manifests/train.jsonl")
        self.assertEqual(str(args.validation_manifest), "data/manifests/validation.jsonl")
        self.assertEqual(str(args.output), "artifacts/checkpoints/ppocrv6_medium_rec_rxhandbd_finetuned")
        self.assertEqual((args.epochs, args.batch_size, args.learning_rate, args.seed), (3, 8, 1e-5, 20261017))


if __name__ == "__main__":
    unittest.main()
