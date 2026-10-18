"""Compare a Flax NNX CTC port against Paddle's shipped inference predictor.

Inputs to both runtimes are the same preprocessed NCHW float32 arrays. This
tool intentionally fails when a pretrained NNX checkpoint is not supplied:
random-weight shape checks are not model parity.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from PIL import Image

from ocrax.infrastructure.data.images import resize_rec_image


def load_paddle_predictor(model_dir: Path):
    import paddle

    config = paddle.inference.Config(
        str(model_dir / "inference.json"), str(model_dir / "inference.pdiparams")
    )
    config.disable_gpu()
    config.disable_glog_info()
    return paddle.inference.create_predictor(config)


def paddle_forward(predictor, nchw: np.ndarray) -> np.ndarray:
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    if len(input_names) != 1 or len(output_names) != 1:
        raise ValueError(f"Expected one input and output, got {input_names}, {output_names}")
    handle = predictor.get_input_handle(input_names[0])
    handle.reshape(nchw.shape)
    handle.copy_from_cpu(nchw.astype(np.float32, copy=False))
    predictor.run()
    return predictor.get_output_handle(output_names[0]).copy_to_cpu()


def _load_paddle_logits_predictor(model_dir: Path):
    """Expose the pre-softmax CTC logits from the shipped PIR graph."""
    import paddle
    from ocrax.infrastructure.paddle.probe import add_fetches

    temp_dir = Path(tempfile.mkdtemp(prefix="ppocrv6-paddle-logits-"))
    shutil.copy2(model_dir / "inference.pdiparams", temp_dir / "inference.pdiparams")
    add_fetches(model_dir / "inference.json", temp_dir / "inference.json", [(612, "ctc_logits")])
    config = paddle.inference.Config(
        str(temp_dir / "inference.json"), str(temp_dir / "inference.pdiparams")
    )
    config.disable_gpu()
    config.disable_glog_info()
    return paddle.inference.create_predictor(config), temp_dir


def _paddle_forward_outputs(predictor, nchw: np.ndarray) -> dict[str, np.ndarray]:
    input_names = predictor.get_input_names()
    if len(input_names) != 1:
        raise ValueError(f"Expected one Paddle input, got {input_names}")
    handle = predictor.get_input_handle(input_names[0])
    handle.reshape(nchw.shape)
    handle.copy_from_cpu(nchw.astype(np.float32, copy=False))
    predictor.run()
    return {
        name: predictor.get_output_handle(name).copy_to_cpu()
        for name in predictor.get_output_names()
    }




def compare(model_dir: Path, checkpoint: Path, image_paths: list[Path], *, atol: float, rtol: float, report_path: Path | None = None):
    from ocrax.infrastructure.checkpoints.orbax import restore_nnx
    model = restore_nnx(checkpoint)
    predictor = load_paddle_predictor(model_dir)
    logits_predictor, logits_temp_dir = _load_paddle_logits_predictor(model_dir)
    prepared = [resize_rec_image(Image.open(path)) for path in image_paths]
    samples = [item[0] for item in prepared]
    resized_widths = [int(item[1]) for item in prepared]
    batch = np.stack(samples).astype(np.float32)
    paddle_probs = paddle_forward(predictor, batch)
    paddle_logits = _paddle_forward_outputs(logits_predictor, batch)["ctc_logits"]
    jax_intermediates = model.forward_intermediates(jnp.asarray(batch))
    jax_logits = np.asarray(jax_intermediates["ctc_logits"])
    jax_probs = np.asarray(jax_intermediates["probabilities"])
    shutil.rmtree(logits_temp_dir, ignore_errors=True)
    if paddle_probs.shape != jax_probs.shape:
        raise AssertionError(f"Output shape mismatch: Paddle {paddle_probs.shape}, JAX {jax_probs.shape}")
    diff = np.abs(paddle_probs.astype(np.float32) - jax_probs.astype(np.float32))
    relative = diff / np.maximum(np.abs(paddle_probs.astype(np.float32)), 1e-8)
    paddle_ids, jax_ids = paddle_probs.argmax(-1), jax_probs.argmax(-1)
    report = {
        "images": [str(p) for p in image_paths],
        "preprocessing": {
            "input_shape_nchw": list(batch.shape),
            "resized_widths_before_padding": resized_widths,
            "batch_sha256": hashlib.sha256(batch.tobytes()).hexdigest(),
        },
        "shape": list(jax_probs.shape),
        "ctc_logits": {
            "shape": list(jax_logits.shape),
            "max_abs_error": float(np.max(np.abs(paddle_logits - jax_logits))),
            "mean_abs_error": float(np.mean(np.abs(paddle_logits - jax_logits))),
            "max_relative_error": float(np.max(
                np.abs(paddle_logits - jax_logits) / np.maximum(np.abs(paddle_logits), 1e-8)
            )),
            "allclose": bool(np.allclose(paddle_logits, jax_logits, atol=atol, rtol=rtol)),
        },
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "max_relative_error": float(relative.max()),
        "argmax_agreement": float(np.mean(paddle_ids == jax_ids)),
        "mismatched_timestep_count": int(np.count_nonzero(paddle_ids != jax_ids)),
        "timestep_count": int(np.prod(paddle_probs.shape[:2])),
        "per_image_argmax_agreement": {
            str(path): float(np.mean(paddle_ids[i] == jax_ids[i]))
            for i, path in enumerate(image_paths)
        },
        "max_top1_probability_error": float(np.max(np.abs(
            np.max(paddle_probs, axis=-1) - np.max(jax_probs, axis=-1)
        ))),
        "atol": atol,
        "rtol": rtol,
        "allclose": bool(np.allclose(paddle_probs, jax_probs, atol=atol, rtol=rtol)),
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["allclose"] or not report["ctc_logits"]["allclose"]:
        raise SystemExit(2)
    return report
