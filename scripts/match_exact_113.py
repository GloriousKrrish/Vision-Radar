import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import numpy as np
from yolox_torch_transfer import YOLOXNanoTorch

def test_exact_113_match():
    onnx_path = "data/models/yolox_nano.onnx"
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    initializers = {w.name: onnx.numpy_helper.to_array(w) for w in graph.initializer}
    conv_nodes = [n for n in graph.node if n.op_type == 'Conv']

    py_model = YOLOXNanoTorch(num_classes=4)
    py_model.eval()

    py_convs = [m for m in py_model.modules() if isinstance(m, nn.Conv2d)]

    onnx_list = []
    for n in conv_nodes:
        w_arr = initializers[n.input[1]].copy()
        b_arr = initializers[n.input[2]].copy() if len(n.input) > 2 and n.input[2] in initializers else None
        onnx_list.append((w_arr, b_arr, n.name))

    print(f"ONNX Conv nodes: {len(onnx_list)}, PyTorch Conv modules: {len(py_convs)}")

    coco_indices = [5, 2, 3, 7]

    mismatches = 0
    with torch.no_grad():
        for i in range(len(py_convs)):
            w_arr, b_arr, name = onnx_list[i]
            p_mod = py_convs[i]
            p_shape = tuple(p_mod.weight.shape)
            o_shape = w_arr.shape

            if p_shape == o_shape:
                p_mod.weight.copy_(torch.from_numpy(w_arr))
                if p_mod.bias is not None and b_arr is not None:
                    p_mod.bias.copy_(torch.from_numpy(b_arr))
            elif p_shape[0] == 4 and o_shape[0] == 80 and p_shape[1:] == o_shape[1:]:
                w_arr_sub = w_arr[coco_indices]
                p_mod.weight.copy_(torch.from_numpy(w_arr_sub))
                if p_mod.bias is not None and b_arr is not None:
                    b_arr_sub = b_arr[coco_indices]
                    p_mod.bias.copy_(torch.from_numpy(b_arr_sub))
            else:
                mismatches += 1
                print(f"Mismatch at index {i:3d}: PyTorch {p_shape} vs ONNX {o_shape} ({name})")

    print(f"Total shape mismatches in direct graph order: {mismatches}")

if __name__ == "__main__":
    test_exact_113_match()
