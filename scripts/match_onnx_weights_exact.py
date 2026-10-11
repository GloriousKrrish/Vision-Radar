import onnx
import torch
import torch.nn as nn
import numpy as np
from yolox_torch_transfer import YOLOXNanoTorch

def match_exact():
    onnx_model = onnx.load("data/models/yolox_nano.onnx")
    graph = onnx_model.graph
    initializers = {w.name: onnx.numpy_helper.to_array(w) for w in graph.initializer}
    conv_nodes = [n for n in graph.node if n.op_type == 'Conv']

    py_model = YOLOXNanoTorch(num_classes=4)
    py_convs = [m for m in py_model.modules() if isinstance(m, nn.Conv2d)]

    # Map ONNX nodes to py_convs by shape matching queue
    onnx_tensors = []
    for n in conv_nodes:
        w_name = n.input[1]
        w_arr = initializers[w_name]
        b_name = n.input[2] if len(n.input) > 2 else None
        b_arr = initializers[b_name] if b_name else None
        onnx_tensors.append((w_arr, b_arr, n.name))

    print(f"Total ONNX Conv Tensors: {len(onnx_tensors)}, PyTorch Conv Modules: {len(py_convs)}")

    # Let's verify shape distribution
    onnx_shapes = [t[0].shape for t in onnx_tensors]
    py_shapes = [tuple(p.weight.shape) for p in py_convs]

    from collections import Counter
    print("ONNX shape counts:", Counter(onnx_shapes))
    print("PyTorch shape counts:", Counter(py_shapes))

if __name__ == "__main__":
    match_exact()
