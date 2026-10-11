import onnx
import torch
import numpy as np

def inspect_graph():
    onnx_model = onnx.load("data/models/yolox_nano.onnx")
    graph = onnx_model.graph
    initializers = {w.name: onnx.numpy_helper.to_array(w) for w in graph.initializer}
    
    print("Graph node count:", len(graph.node))
    conv_nodes = [n for n in graph.node if n.op_type == 'Conv']
    bn_nodes = [n for n in graph.node if n.op_type == 'BatchNormalization']
    print(f"Total Conv nodes: {len(conv_nodes)}, Total BN nodes: {len(bn_nodes)}")
    
    for i, n in enumerate(conv_nodes[:10]):
        w_name = n.input[1]
        w_arr = initializers.get(w_name)
        b_name = n.input[2] if len(n.input) > 2 else None
        b_arr = initializers.get(b_name) if b_name else None
        print(f"Conv {i}: name={n.name}, weight_shape={w_arr.shape if w_arr is not None else None}, bias={b_arr is not None}")

if __name__ == "__main__":
    inspect_graph()
