"""Create a PIR inference JSON that exposes selected SSA values as outputs."""

from __future__ import annotations

import json
from pathlib import Path

import paddle
from paddle.base import core
from paddle.static.pir_io import append_pir_fetch_ops


def add_fetches(model_json: Path, output_json: Path, probes: list[tuple[int, str]]) -> None:
    doc = json.loads(model_json.read_text(encoding="utf-8"))
    json_ops = doc["program"]["regions"][0]["blocks"][0]["ops"]
    paddle.enable_static()
    program = paddle.static.Program()
    if not core.deserialize_pir_program(str(model_json), program):
        raise RuntimeError(f"Could not deserialize Paddle PIR program: {model_json}")

    # Map serialized SSA IDs to the matching PIR Values by operation/result order.
    values = {}
    for op_index, json_op in enumerate(json_ops):
        if op_index >= len(program.global_block().ops):
            break
        pir_op = program.global_block().ops[op_index]
        for result_index, result in enumerate(json_op.get("O", [])):
            if isinstance(result, dict) and isinstance(result.get("%"), int):
                values[result["%"]] = pir_op.results()[result_index]

    fetch_ops = [op for op in program.global_block().ops if op.name() == "pd_op.fetch"]
    original_fetches = []
    for json_op in json_ops:
        if json_op.get("#") != "1.fetch":
            continue
        name = next(a["AT"]["D"] for a in json_op["A"] if a.get("N") == "name")
        source_id = json_op["I"][0]["%"]
        original_fetches.append((values[source_id], name))
    for op in fetch_ops:
        op.get_parent_block().remove_op(op)

    names = {name for _, name in original_fetches}
    for value_id, name in probes:
        if value_id not in values:
            raise KeyError(f"PIR value %{value_id} does not exist")
        if name in names:
            raise ValueError(f"Duplicate fetch name: {name}")
        original_fetches.append((values[value_id], name))
        names.add(name)
    with paddle.static.program_guard(program):
        append_pir_fetch_ops(program, original_fetches)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    core.serialize_pir_program(program, str(output_json), readable=True)
