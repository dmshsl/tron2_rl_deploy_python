"""Promote MLP arithmetic to FP64, keeping the deployment FLOAT32 interface.

Usage: python stabilize_onnx.py --source policy.onnx --out stable-policy.onnx
Requires onnx only at export time; inference still needs only onnxruntime.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


class ModelConversionError(ValueError):
    pass


def stabilize(source: Path, out: Path) -> None:
    model = onnx.load(source)
    onnx.checker.check_model(model)
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ModelConversionError("expected one input and one output")
    if any(node.op_type not in {"MatMul", "Add", "Elu"} for node in graph.node):
        raise ModelConversionError("expected an unconverted MatMul/Add/Elu MLP")
    if any(info.type.tensor_type.elem_type != TensorProto.FLOAT for info in (*graph.input, *graph.output)):
        raise ModelConversionError("expected FLOAT32 model interface")
    input_name, output_name = graph.input[0].name, graph.output[0].name
    for tensor in graph.initializer:
        if tensor.data_type != TensorProto.FLOAT:
            raise ModelConversionError(f"expected FLOAT32 initializer: {tensor.name}")
        tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float64), tensor.name))
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == input_name:
                node.input[index] = input_name + "_fp64"
        for index, name in enumerate(node.output):
            if name == output_name:
                node.output[index] = output_name + "_fp64"
    for info in graph.value_info:
        info.type.tensor_type.elem_type = TensorProto.DOUBLE
    nodes = []
    for index, node in enumerate(graph.node):
        if node.op_type != "Elu":
            nodes.append(node)
            continue
        if any(attribute.name != "alpha" or attribute.f != 1. for attribute in node.attribute):
            raise ModelConversionError("only alpha=1 ELU is supported")
        prefix = f"stable_elu_{index}"
        for name, value in (("zero", 0.), ("one", 1.)):
            graph.initializer.append(numpy_helper.from_array(np.array(value, dtype=np.float64), prefix + name))
        nodes.extend((
            helper.make_node("Greater", [node.input[0], prefix + "zero"], [prefix + "positive"]),
            helper.make_node("Min", [node.input[0], prefix + "zero"], [prefix + "negative"]),
            helper.make_node("Exp", [prefix + "negative"], [prefix + "exp"]),
            helper.make_node("Sub", [prefix + "exp", prefix + "one"], [prefix + "expm1"]),
            helper.make_node("Where", [prefix + "positive", node.input[0], prefix + "expm1"], list(node.output)),
        ))
    del graph.node[:]
    graph.node.extend(nodes)
    graph.node.insert(0, helper.make_node("Cast", [input_name], [input_name + "_fp64"], to=TensorProto.DOUBLE))
    graph.node.append(helper.make_node("Cast", [output_name + "_fp64"], [output_name], to=TensorProto.FLOAT))
    helper.set_model_props(model, {"tron2.internal_precision": "float64",
                                  "tron2.source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    onnx.checker.check_model(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    stabilize(args.source, args.out)
