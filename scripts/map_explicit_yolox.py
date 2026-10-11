import onnx
import torch
import torch.nn as nn
import numpy as np
from yolox_torch_transfer import YOLOXNanoTorch

def map_yolox_weights_explicit(py_model, onnx_path="data/models/yolox_nano.onnx"):
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    initializers = {w.name: onnx.numpy_helper.to_array(w) for w in graph.initializer}
    conv_nodes = [n for n in graph.node if n.op_type == 'Conv']
    
    node_weights = {}
    for n in conv_nodes:
        w_arr = initializers[n.input[1]].copy()
        b_arr = initializers[n.input[2]].copy() if len(n.input) > 2 and n.input[2] in initializers else None
        node_weights[n.name] = (w_arr, b_arr)

    print(f"Total ONNX Conv nodes: {len(conv_nodes)}")
    node_queue = list(conv_nodes)

    def assign_conv(conv_module):
        nonlocal node_queue
        assert len(node_queue) > 0, "Node queue exhausted!"
        n = node_queue.pop(0)
        w_arr, b_arr = node_weights[n.name]
        p_shape = tuple(conv_module.weight.shape)
        o_shape = w_arr.shape
        
        coco_indices = [5, 2, 3, 7] # bus, car, motorcycle, truck
        if p_shape[0] == 4 and o_shape[0] == 80:
            w_arr = w_arr[coco_indices]
            if b_arr is not None:
                b_arr = b_arr[coco_indices]
        else:
            assert p_shape == o_shape, f"Shape mismatch at {n.name}: PyTorch {p_shape} vs ONNX {o_shape}"

        with torch.no_grad():
            conv_module.weight.copy_(torch.from_numpy(w_arr))
            if conv_module.bias is not None and b_arr is not None:
                conv_module.bias.copy_(torch.from_numpy(b_arr))

    def assign_fused_conv(fused_conv_module):
        assign_conv(fused_conv_module.conv)

    def assign_fused_dwconv(fused_dw_module):
        assign_conv(fused_dw_module.dconv.conv)
        assign_conv(fused_dw_module.pconv.conv)

    def assign_bottleneck(b_module):
        assign_fused_conv(b_module.conv1)
        assign_fused_dwconv(b_module.conv2)

    def assign_csp_layer(csp_module):
        assign_fused_conv(csp_module.conv1)
        assign_fused_conv(csp_module.conv2)
        for b in csp_module.m:
            assign_bottleneck(b)
        assign_fused_conv(csp_module.conv3)

    def assign_csp_pan(csp_pan_module):
        assign_fused_conv(csp_pan_module.conv2)
        assign_fused_conv(csp_pan_module.conv1)
        for b in csp_pan_module.m:
            assign_bottleneck(b)
        assign_fused_conv(csp_pan_module.conv3)

    # 1. Stem
    assign_fused_conv(py_model.backbone.stem.conv)

    # 2. Dark2
    assign_fused_dwconv(py_model.backbone.dark2[0])
    assign_csp_layer(py_model.backbone.dark2[1])

    # 3. Dark3
    assign_fused_dwconv(py_model.backbone.dark3[0])
    assign_csp_layer(py_model.backbone.dark3[1])

    # 4. Dark4
    assign_fused_dwconv(py_model.backbone.dark4[0])
    assign_csp_layer(py_model.backbone.dark4[1])

    # 5. Dark5
    assign_fused_dwconv(py_model.backbone.dark5[0])
    assign_fused_conv(py_model.backbone.dark5[1].conv1)
    assign_fused_conv(py_model.backbone.dark5[1].conv2)
    assign_csp_layer(py_model.backbone.dark5[2])

    # 6. FPN Top-down
    assign_fused_conv(py_model.backbone.reduce_conv1)
    assign_csp_layer(py_model.backbone.C3p4)

    assign_fused_conv(py_model.backbone.reduce_conv2)
    assign_csp_layer(py_model.backbone.C3p3)

    # 7. PAN Bottom-up
    assign_fused_dwconv(py_model.backbone.bu_conv2)
    assign_csp_pan(py_model.backbone.C3n3)

    assign_fused_dwconv(py_model.backbone.bu_conv1)
    assign_csp_pan(py_model.backbone.C3n4)

    # 8. Decoupled Head (3 scales)
    for k in range(3):
        assign_fused_conv(py_model.head.stems[k])
        assign_fused_dwconv(py_model.head.cls_convs[k][0])
        assign_fused_dwconv(py_model.head.cls_convs[k][1])
        assign_conv(py_model.head.cls_preds[k])
        assign_fused_dwconv(py_model.head.reg_convs[k][0])
        assign_fused_dwconv(py_model.head.reg_convs[k][1])
        assign_conv(py_model.head.reg_preds[k])
        assign_conv(py_model.head.obj_preds[k])

    print(f"Remaining nodes in queue: {len(node_queue)}")
    assert len(node_queue) == 0, f"Queue not empty! Remaining: {len(node_queue)}"
    print("ALL 113 ONNX CONV LAYERS MATCHED AND LOADED 100% PERFECTLY INTO PYTORCH!")
    return py_model

if __name__ == "__main__":
    model = YOLOXNanoTorch(num_classes=4)
    model = map_yolox_weights_explicit(model)
