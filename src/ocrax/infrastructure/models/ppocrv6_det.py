"""Flax NNX PP-OCRv6 medium text detector (DBNet++ inference path).

Executes the PP-OCRv6 medium detection architecture in JAX with full
mathematical parity to the Paddle PIR model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from flax import nnx

GRAPH_FILE = Path(__file__).parent / "ppocrv6_det_graph.json"
SHAPES_FILE = Path(__file__).parent / "ppocrv6_det_shapes.json"


def _load_graph_plan() -> list[Any]:
    if not GRAPH_FILE.is_file():
        raise FileNotFoundError(f"Detection graph file not found: {GRAPH_FILE}")
    return json.loads(GRAPH_FILE.read_text(encoding="utf-8"))


def _evaluate_plan(plan: list[Any], params: dict[str, Any], image_nchw: jax.Array) -> jax.Array:
    """Execute the compiled JAX forward pass."""
    env: dict[Any, Any] = {}
    for item in plan:
        tag = item[0]
        if tag == "p":
            env[item[1]] = params[item[2]]
        elif tag == "data":
            env[item[1]] = image_nchw
        elif tag == "const":
            env[item[1]] = item[2]
        elif tag == "full":
            env[item[1]] = jnp.full(item[3], item[2], dtype=jnp.float32)
        elif tag == "conv":
            res = lax.conv_general_dilated(
                env[item[2]],
                env[item[3]],
                window_strides=item[4],
                padding=item[5],
                lhs_dilation=(1, 1),
                rhs_dilation=item[6],
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
                feature_group_count=item[7],
            )
            env[item[1]] = res
        elif tag == "conv_trans":
            in_w = env[item[3]]
            kh, kw = in_w.shape[2], in_w.shape[3]
            pad = (
                (kh - 1 - item[5][0], kh - 1 - item[5][0]),
                (kw - 1 - item[5][1], kw - 1 - item[5][1]),
            )
            w_trans = jnp.transpose(in_w, (1, 0, 2, 3))[:, :, ::-1, ::-1]
            res = lax.conv_general_dilated(
                env[item[2]],
                w_trans,
                window_strides=(1, 1),
                padding=pad,
                lhs_dilation=item[4],
                rhs_dilation=item[6],
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
                feature_group_count=item[7],
            )
            env[item[1]] = res
        elif tag == "add":
            env[item[1]] = env[item[2]] + env[item[3]]
        elif tag == "mul":
            env[item[1]] = env[item[2]] * env[item[3]]
        elif tag == "relu":
            env[item[1]] = jax.nn.relu(env[item[2]])
        elif tag == "gelu":
            env[item[1]] = jax.nn.gelu(env[item[2]], approximate=False)
        elif tag == "hsigmoid":
            env[item[1]] = jnp.clip((env[item[2]] + 3.0) / 6.0, 0.0, 1.0)
        elif tag == "sigmoid":
            env[item[1]] = jax.nn.sigmoid(env[item[2]])
        elif tag == "interp":
            x_in = env[item[2]]
            scale = item[3]
            if scale:
                sh = int(scale) if isinstance(scale, (int, float)) else int(scale[0])
                sw = int(scale) if isinstance(scale, (int, float)) else int(scale[1])
                new_shape = (x_in.shape[0], x_in.shape[1], x_in.shape[2] * sh, x_in.shape[3] * sw)
            else:
                new_shape = (x_in.shape[0], x_in.shape[1], item[4], item[5])
            env[item[1]] = jax.image.resize(x_in, shape=new_shape, method="nearest")
        elif tag == "pool":
            x_in = env[item[2]]
            if item[4] or item[3] == "avg":
                env[item[1]] = jnp.mean(x_in, axis=(2, 3), keepdims=True)
            else:
                env[item[1]] = lax.reduce_window(
                    x_in,
                    -jnp.inf,
                    lax.max,
                    window_dimensions=(1, 1, item[5][0], item[5][1]),
                    window_strides=(1, 1, item[6][0], item[6][1]),
                    padding="SAME",
                )
        elif tag == "mean":
            env[item[1]] = jnp.mean(env[item[2]], axis=item[3], keepdims=item[4])
        elif tag == "bn":
            x_in = env[item[2]]
            mean = env[item[3]].reshape(1, -1, 1, 1)
            var = env[item[4]].reshape(1, -1, 1, 1)
            scale = env[item[5]].reshape(1, -1, 1, 1)
            bias = env[item[6]].reshape(1, -1, 1, 1)
            env[item[1]] = (x_in - mean) / jnp.sqrt(var + item[7]) * scale + bias
        elif tag == "reshape":
            shape = env[item[3]] if item[3] is not None else item[4]
            env[item[1]] = jnp.reshape(env[item[2]], shape)
        elif tag == "combine":
            env[item[1]] = [env[k] for k in item[2]]
        elif tag == "concat":
            env[item[1]] = jnp.concatenate(env[item[2]], axis=item[3])
        elif tag == "fetch":
            return env[item[1]]

    raise RuntimeError("Graph execution did not reach a fetch operation")


class PP_OCRv6MediumDet(nnx.Module):
    """PP-OCRv6 medium text detection inference graph in Flax NNX."""

    def __init__(self, initial_params: dict[str, Any] | None = None, rngs: nnx.Rngs | None = None):
        self._plan = _load_graph_plan()
        
        # Collect required parameter names from the plan
        param_names = [item[2] for item in self._plan if item[0] == "p"]
        
        if initial_params is not None:
            self.params = nnx.Dict(
                {name: nnx.Param(jnp.asarray(initial_params[name], dtype=jnp.float32)) for name in param_names}
            )
        else:
            shapes = json.loads(SHAPES_FILE.read_text(encoding="utf-8")) if SHAPES_FILE.is_file() else {}
            self.params = nnx.Dict(
                {name: nnx.Param(jnp.zeros(tuple(shapes.get(name, (1,))), dtype=jnp.float32)) for name in param_names}
            )

    def __call__(self, images_nchw: jax.Array) -> jax.Array:
        """Run text detection.
        
        Args:
            images_nchw: Batch of normalized image tensors with shape (B, 3, H, W).
            
        Returns:
            Probability heatmap with shape (B, 1, H, W) containing values in [0, 1].
        """
        # Convert nnx.Dict of nnx.Param into flat parameter dictionary for evaluate_plan
        param_dict = {k: v.get_value() for k, v in self.params.items()}
        return _evaluate_plan(self._plan, param_dict, images_nchw)
