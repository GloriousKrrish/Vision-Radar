import onnx
import torch
import torch.nn as nn
from yolox_torch_transfer import YOLOXNanoTorch

def debug_shapes():
    onnx_model = onnx.load("data/models/yolox_nano.onnx")
    graph = onnx_model.graph
    initializers = {w.name: onnx.numpy_helper.to_array(w) for w in graph.initializer}
    conv_nodes = [n for n in graph.node if n.op_type == 'Conv']

    py_model = YOLOXNanoTorch(num_classes=4)
    py_convs = [m for m in py_model.modules() if isinstance(m, nn.Conv2d)]

    for idx, (c_node, p_module) in enumerate(zip(conv_nodes, py_convs)):
        w_arr = initializers[c_node.input[1]]
        p_shape = tuple(p_module.weight.shape)
        o_shape = tuple(w_arr.shape)
        if p_shape != o_shape and not (p_module.out_channels == 4 and o_shape[0] == 80):
            print(f"Mismatch at index {idx:3d}: ONNX {o_shape} vs PyTorch {p_shape}")

if __name__ == "__main__":
    debug_shapes()
