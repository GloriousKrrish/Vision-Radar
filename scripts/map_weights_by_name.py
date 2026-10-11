import onnx
import torch
import torch.nn as nn
import numpy as np
from yolox_torch_transfer import YOLOXNanoTorch

def map_by_shape_queue(pytorch_model, onnx_path="data/models/yolox_nano.onnx"):
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    initializers = {w.name: onnx.numpy_helper.to_array(w) for w in graph.initializer}
    conv_nodes = [n for n in graph.node if n.op_type == 'Conv']

    # Extract all conv weights from ONNX
    onnx_weights = []
    for n in conv_nodes:
        w_arr = initializers[n.input[1]].copy()
        b_arr = initializers[n.input[2]].copy() if len(n.input) > 2 and n.input[2] in initializers else None
        onnx_weights.append((w_arr, b_arr, n.name))

    # Match ONNX weights to PyTorch model by traversing PyTorch execution graph order or shape queue
    # PyTorch modules
    py_convs = [m for m in pytorch_model.modules() if isinstance(m, nn.Conv2d)]
    
    # Create shape-indexed pool of ONNX weights
    shape_pool = {}
    for w_arr, b_arr, name in onnx_weights:
        s = tuple(w_arr.shape)
        if s not in shape_pool:
            shape_pool[s] = []
        shape_pool[s].append((w_arr, b_arr, name))

    print(f"ONNX total weights: {len(onnx_weights)}, PyTorch total convs: {len(py_convs)}")

    coco_indices = [5, 2, 3, 7] # bus, car, motorcycle, truck

    success_count = 0
    with torch.no_grad():
        for idx, p_module in enumerate(py_convs):
            target_shape = tuple(p_module.weight.shape)
            
            # Special case for output class heads: 80 COCO classes -> 4 classes
            if target_shape[0] == 4 and target_shape[1] == 64 and target_shape[2] == 1 and target_shape[3] == 1:
                # Find matching (80, 64, 1, 1) in ONNX
                if (80, 64, 1, 1) in shape_pool and len(shape_pool[(80, 64, 1, 1)]) > 0:
                    w_arr, b_arr, name = shape_pool[(80, 64, 1, 1)].pop(0)
                    w_arr = w_arr[coco_indices]
                    if b_arr is not None:
                        b_arr = b_arr[coco_indices]
                    p_module.weight.copy_(torch.from_numpy(w_arr))
                    if p_module.bias is not None and b_arr is not None:
                        p_module.bias.copy_(torch.from_numpy(b_arr))
                    success_count += 1
                    continue

            if target_shape in shape_pool and len(shape_pool[target_shape]) > 0:
                w_arr, b_arr, name = shape_pool[target_shape].pop(0)
                p_module.weight.copy_(torch.from_numpy(w_arr))
                if p_module.bias is not None and b_arr is not None:
                    p_module.bias.copy_(torch.from_numpy(b_arr))
                success_count += 1
            else:
                print(f"Warning: No match for PyTorch layer {idx} shape {target_shape}")

    print(f"Successfully loaded {success_count} / {len(py_convs)} weights into PyTorch YOLOX-Nano model!")
    return pytorch_model

if __name__ == "__main__":
    model = YOLOXNanoTorch(num_classes=4)
    model = map_by_shape_queue(model)
    dummy_input = torch.zeros(1, 3, 640, 640)
    out = model(dummy_input)
    print("PyTorch forward pass out shape:", out.shape)
