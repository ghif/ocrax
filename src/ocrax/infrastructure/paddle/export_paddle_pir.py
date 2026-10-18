"""Export named arrays from a Paddle 3.x PIR inference model's .pdiparams."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import paddle
from paddle.base import core


def export(model_dir: Path, output: Path) -> dict:
    paddle.enable_static()
    program = paddle.static.Program()
    model_file = model_dir / "inference.json"
    params_file = model_dir / "inference.pdiparams"
    if not model_file.is_file() or not params_file.is_file():
        raise FileNotFoundError(f"Expected inference.json and inference.pdiparams in {model_dir}")
    if not core.deserialize_pir_program(str(model_file), program):
        raise RuntimeError(f"Could not deserialize Paddle PIR program: {model_file}")
    executor = paddle.static.Executor(paddle.CPUPlace())
    paddle.static.load_vars(
        executor, dirname=str(model_dir), main_program=program,
        filename=params_file.name,
    )
    scope = paddle.static.global_scope()
    arrays = {}
    for variable in program.list_vars():
        if not (variable.is_parameter or variable.get_defining_op().name() == "builtin.parameter"):
            continue
        scoped = scope.find_var(variable.name)
        if scoped is None:
            raise RuntimeError(f"Paddle failed to load parameter {variable.name}")
        value = np.array(scoped.get_tensor())
        if value.size == 0:
            raise RuntimeError(f"Paddle loaded an empty parameter: {variable.name}")
        arrays[variable.name] = value
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    manifest = {
        "model_dir": str(model_dir.resolve()),
        "arrays_file": str(output.resolve()),
        "parameter_count": len(arrays),
        "parameters": {name: list(value.shape) for name, value in arrays.items()},
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
