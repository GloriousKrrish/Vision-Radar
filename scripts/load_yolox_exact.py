import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import numpy as np

# We instantiate PyTorch YOLOX-Nano model with matching layer channels:
from yolox_torch_transfer import FusedConv, FusedDWConv, Focus, YOLOXNanoTorch

def create_exact_yolox():
    model = YOLOXNanoTorch(num_classes=4)
    # Let's adjust channel sizes for PAFPN blocks to match YOLOX-Nano ONNX:
    # P3: 64, P4: 128, P5: 128
    model.backbone.dark5 = nn.Sequential(
        FusedDWConv(128, 256, ksize=3, stride=2), # (256, 128)
        # SPPBottleneck
        nn.Sequential(
            FusedConv(256, 128, ksize=1, stride=1),
            FusedConv(512, 256, ksize=1, stride=1),
        ),
        # CSPLayer dark5
        nn.Sequential(
            FusedConv(256, 128, ksize=1, stride=1),
            FusedConv(256, 128, ksize=1, stride=1),
            # bottleneck
            FusedConv(128, 128, ksize=1, stride=1),
            FusedDWConv(128, 128, ksize=3, stride=1),
            FusedConv(256, 256, ksize=1, stride=1),
        )
    )
    return model

def test_onnx_pytorch_alignment():
    onnx_path = "data/models/yolox_nano.onnx"
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    initializers = {w.name: onnx.numpy_helper.to_array(w) for w in graph.initializer}
    conv_nodes = [n for n in graph.node if n.op_type == 'Conv']

    print(f"Total ONNX Conv nodes: {len(conv_nodes)}")

    # Let's create an ONNX Runtime Session
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Test dummy image input
    np.random.seed(42)
    dummy_img = np.random.randn(1, 3, 640, 640).astype(np.float32)

    # Run ONNX inference
    onnx_out = session.run(None, {"images": dummy_img})[0]
    print("ONNX output shape:", onnx_out.shape)

if __name__ == "__main__":
    test_onnx_pytorch_alignment()
