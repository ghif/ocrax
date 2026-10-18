"""Convert the exported PP-OCRv6 medium inference tensors into an Orbax NNX state.

The mapping is tied to the shipped medium-recognition inference graph, whose
PPLCNetV4 RepDW and ConvBN blocks have been fused for inference. It is not a
generic Paddle checkpoint converter.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from ocrax.infrastructure.models.ppocrv6 import PP_OCRv6MediumRec
from ocrax.infrastructure.checkpoints.orbax import save_state


def _conv_key(layer: int, field: str) -> str:
    return f"conv2d_{layer}.{field}_0"


def _as_hwio(weight: np.ndarray) -> np.ndarray:
    if weight.ndim != 4:
        raise ValueError(f"Expected Paddle OIHW convolution tensor, got {weight.shape}")
    return np.transpose(weight, (2, 3, 1, 0))


def convert(arrays_path: Path, output_dir: Path) -> dict:
    arrays_file = np.load(arrays_path)
    arrays = {key: arrays_file[key] for key in arrays_file.files}
    model = PP_OCRv6MediumRec()
    _, state = nnx.split(model)
    flat = dict(nnx.to_flat_state(state))
    used: set[str] = set()

    def assign(path: tuple, value: np.ndarray, source: str):
        if path not in flat:
            raise KeyError(f"NNX path does not exist: {'/'.join(map(str, path))}")
        var = flat[path]
        expected = tuple(var.shape)
        value = np.asarray(value)
        if value.shape != expected:
            raise ValueError(f"{source} -> {'/'.join(map(str, path))}: {value.shape} != {expected}")
        var.value = jnp.asarray(value)
        used.add(source)

    def get(key: str) -> np.ndarray:
        if key not in arrays:
            raise KeyError(f"Missing Paddle tensor: {key}")
        used.add(key)
        return arrays[key]

    def conv(path: tuple, layer: int):
        kernel = _as_hwio(get(_conv_key(layer, "w")))
        assign(path + ("conv", "kernel"), kernel, _conv_key(layer, "w"))
        bias_key = _conv_key(layer, "b")
        bias = get(bias_key) if bias_key in arrays else np.zeros(kernel.shape[-1], np.float32)
        assign(path + ("conv", "bias"), bias, bias_key if bias_key in arrays else "synthetic-zero-bias")
        return kernel

    def identity_bn(path: tuple):
        channels = flat[path + ("bn", "scale")].shape[0]
        eps = 1e-5
        assign(path + ("bn", "scale"), np.ones(channels, np.float32), "identity-bn")
        assign(path + ("bn", "bias"), np.zeros(channels, np.float32), "identity-bn")
        assign(path + ("bn", "mean"), np.zeros(channels, np.float32), "identity-bn")
        assign(path + ("bn", "var"), np.full(channels, 1.0 - eps, np.float32), "identity-bn")

    def fused_convbn(path: tuple, layer: int):
        conv(path, layer)
        identity_bn(path)

    def repdw(path: tuple, layer: int):
        kernel = _as_hwio(get(_conv_key(layer, "w")))
        # The Paddle inference graph fuses the RepDW branches and the trailing
        # identity BatchNorm. Keep Flax's explicit residual by subtracting it
        # from the stored 3x3 kernel before adding x in the module.
        if kernel.shape[0:2] != (3, 3) or kernel.shape[2] != 1:
            raise ValueError(f"Unexpected reparameterized depthwise kernel {kernel.shape}")
        kernel = kernel.copy()
        channels = kernel.shape[-1]
        kernel[1, 1, 0, :] -= 1.0
        path3 = path + ("branch3",)
        assign(path3 + ("conv", "kernel"), kernel, _conv_key(layer, "w"))
        bias_key = _conv_key(layer, "b")
        bias = get(bias_key) if bias_key in arrays else np.zeros(channels, np.float32)
        assign(path3 + ("conv", "bias"), bias, bias_key if bias_key in arrays else "synthetic-zero-bias")
        identity_bn(path3)
        assign(path + ("branch1", "kernel"), np.zeros_like(flat[path + ("branch1", "kernel")].get_value()), "zero-rep-branch")
        assign(path + ("identity_bn", "scale"), np.ones(channels, np.float32), "identity-bn")
        assign(path + ("identity_bn", "bias"), np.zeros(channels, np.float32), "identity-bn")
        assign(path + ("identity_bn", "mean"), np.zeros(channels, np.float32), "identity-bn")
        assign(path + ("identity_bn", "var"), np.full(channels, 1.0 - 1e-5, np.float32), "identity-bn")

    def se(path: tuple, first: int, second: int):
        for module_name, layer in (("fc1", first), ("fc2", second)):
            base = path + (module_name,)
            wkey, bkey = _conv_key(layer, "w"), _conv_key(layer, "b")
            assign(base + ("kernel",), _as_hwio(get(wkey)), wkey)
            assign(base + ("bias",), get(bkey), bkey)

    # Paddle's backbone inference graph folds each ConvBN and RepDW block into
    # a biased convolution. The explicit NNX BatchNorm nodes are initialized
    # to the identity transform so this representation has the same eval path.
    for stem_name, layer in (("stem1", 73), ("stem2a", 74), ("stem2b", 75), ("stem3", 76), ("stem4", 77)):
        fused_convbn(("backbone", "conv1", stem_name, "block"), layer)

    # Each tuple gives: stage, block, depthwise fused conv, expand, compress,
    # RepDW flag, optional pair of squeeze/excitation 1x1 convolutions.
    blocks = [
        ("blocks2", 0, 79, 80, 81, True, (7, 8)),
        ("blocks3", 0, 82, 83, 84, False, None),
        ("blocks3", 1, 86, 87, 88, True, None),
        ("blocks3", 2, 90, 91, 92, True, (20, 21)),
        ("blocks4", 0, 93, 94, 95, False, None),
        ("blocks4", 1, 97, 98, 99, True, (29, 30)),
        ("blocks4", 2, 101, 102, 103, True, None),
        ("blocks4", 3, 105, 106, 107, True, (39, 40)),
        ("blocks4", 4, 109, 110, 111, True, None),
        ("blocks4", 5, 113, 114, 115, True, (49, 50)),
        ("blocks4", 6, 117, 118, 119, True, None),
        ("blocks5", 0, 120, 121, 122, False, None),
        ("blocks5", 1, 124, 125, 126, True, (62, 63)),
        ("blocks5", 2, 128, 129, 130, True, None),
    ]
    for stage, index, dw_layer, expand_layer, compress_layer, is_repdw, se_layers in blocks:
        root = ("backbone", stage, index)
        if is_repdw:
            repdw(root + ("dw",), dw_layer)
        else:
            fused_convbn(root + ("dw",), dw_layer)
        if se_layers:
            se(root + ("se",), *se_layers)
        fused_convbn(root + ("expand",), expand_layer)
        fused_convbn(root + ("compress",), compress_layer)

    # Neck convolutions are not fused with their BatchNorm in this graph.
    # The serialized graph's SSA dataflow shows conv70/bn58 feeds local_conv,
    # while conv72/bn60 is the residual skip added after SVTR.
    for name, conv_id, bn_id in (("conv_reduce", 70, 58), ("local", 71, 59), ("skip_conv", 72, 60)):
        root = ("neck", name, "block")
        conv(root, conv_id)
        # These three source convolutions have no separate bias; set the extra
        # Flax bias to zero, then import the Paddle normalization statistics.
        for target_name, source_suffix in (("scale", "w_0"), ("mean", "w_1"), ("var", "w_2"), ("bias", "b_0")):
            key = f"batch_norm2d_{bn_id}.{source_suffix}"
            assign(root + ("bn", target_name), get(key), key)

    # SVTR transformer dense and normalization layers.
    for block in range(2):
        linear_base = block * 4
        norm_base = block * 2
        for name, layer in (("qkv", linear_base), ("proj", linear_base + 1), ("fc1", linear_base + 2), ("fc2", linear_base + 3)):
            root = ("neck", "blocks", block, name)
            wkey, bkey = f"linear_{layer}.w_0", f"linear_{layer}.b_0"
            assign(root + ("kernel",), get(wkey), wkey)
            assign(root + ("bias",), get(bkey), bkey)
        for name, layer in (("norm1", norm_base), ("norm2", norm_base + 1)):
            root = ("neck", "blocks", block, name)
            wkey, bkey = f"layer_norm_{layer}.w_0", f"layer_norm_{layer}.b_0"
            assign(root + ("scale",), get(wkey), wkey)
            assign(root + ("bias",), get(bkey), bkey)
    for target_name, source_suffix in (("scale", "w_0"), ("bias", "b_0")):
        key = f"layer_norm_4.{source_suffix}"
        assign(("neck", "norm", target_name), get(key), key)
    for target_name, source_suffix, transpose in (("kernel", "w_0", False), ("bias", "b_0", False)):
        key = f"linear_8.{source_suffix}"
        value = get(key)
        assign(("ctc", target_name), value.T if transpose else value, key)

    source_tensors = {k for k in arrays if k not in {"__metadata__"}}
    consumed = source_tensors & used
    missing_sources = sorted(source_tensors - consumed)
    if missing_sources:
        raise ValueError(f"Unmapped Paddle tensors ({len(missing_sources)}): {missing_sources}")

    nnx.update(model, state)
    save_state(output_dir, state)
    report = {
        "source_arrays": str(arrays_path.resolve()),
        "orbax_checkpoint": str(output_dir.resolve()),
        "source_tensor_count": len(source_tensors),
        "consumed_tensor_count": len(consumed),
        "mapping": "candidate PP-OCRv6 medium inference graph v6 map; all source keys assigned, output parity not verified",
        "notes": [
            "CTC inference branch only; configured NRTR auxiliary training head is absent.",
            "Backbone BN/RepDW operators are represented by identity BN nodes plus fused conv parameters.",
            "This conversion requires numerical parity before any fine-tuning run.",
        ],
    }
    (output_dir / "conversion.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
