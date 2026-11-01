#!/usr/bin/env python3
"""Verify mathematical parity between Paddle PIR and JAX PP-OCRv6 detection models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = Path("/Users/mghifary/.paddlex/official_models/PP-OCRv6_medium_det")
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts/checkpoints/ppocrv6_medium_det_orbax"


def check_parity(
    model_dir: Path, checkpoint: Path, atol: float = 1e-4, rtol: float = 1e-4
) -> dict[str, float]:
    import paddle.inference as paddle_infer
    from ocrax.infrastructure.checkpoints.orbax import restore_det_nnx
    import jax.numpy as jnp

    # Setup Paddle predictor
    config = paddle_infer.Config(
        str(model_dir / "inference.json"), str(model_dir / "inference.pdiparams")
    )
    config.disable_gpu()
    predictor = paddle_infer.create_predictor(config)

    # Setup JAX model
    jax_model = restore_det_nnx(checkpoint)

    # Test on deterministic input
    x = np.linspace(-1.0, 1.0, 1 * 3 * 96 * 96, dtype=np.float32).reshape(1, 3, 96, 96)

    # Paddle inference
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    input_handle.reshape([1, 3, 96, 96])
    input_handle.copy_from_cpu(x)
    predictor.run()
    output_names = predictor.get_output_names()
    paddle_out = predictor.get_output_handle(output_names[0]).copy_to_cpu()

    # JAX inference
    jax_out = np.array(jax_model(jnp.asarray(x)))

    diff = np.abs(jax_out - paddle_out)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    if max_diff > atol:
        raise AssertionError(
            f"Parity check failed: max absolute difference {max_diff:.6e} > {atol:.6e}"
        )

    return {
        "max_difference": max_diff,
        "mean_difference": mean_diff,
        "tolerance": atol,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Paddle inference directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Converted Orbax detection checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Maximum allowable absolute difference (default: 1e-4)",
    )
    args = parser.parse_args()

    results = check_parity(args.model_dir.resolve(), args.checkpoint.resolve(), atol=args.atol)
    print(
        f"PP-OCRv6 Detection Parity Check PASSED! "
        f"Max diff: {results['max_difference']:.6e}, Mean diff: {results['mean_difference']:.6e} "
        f"(atol={results['tolerance']:.6e})"
    )


if __name__ == "__main__":
    main()
