import onnx
import torch
import torch.nn as nn
import numpy as np
from scripts.yolox_nn import BaseConv, DWConv, CSPLayer, SPPBottleneck, YOLOPAFPN, YOLOXHead, YOLOXNano

class FusedConv(nn.Module):
    """Fused Conv + Bias matching ONNX Conv layers directly."""
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=True
        )
        self.act = nn.SiLU(inplace=True) if act == "silu" else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

class FusedDWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = FusedConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act)
        self.pconv = FusedConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        return self.pconv(self.dconv(x))

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, act="silu"):
        super().__init__()
        self.conv = FusedConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)

def check_mapping():
    onnx_model = onnx.load("data/models/yolox_nano.onnx")
    graph = onnx_model.graph
    initializers = {w.name: onnx.numpy_helper.to_array(w) for w in graph.initializer}
    conv_nodes = [n for n in graph.node if n.op_type == 'Conv']
    print(f"Total ONNX Conv nodes: {len(conv_nodes)}")
    for idx, n in enumerate(conv_nodes[:15]):
        w_shape = initializers[n.input[1]].shape
        print(f"ONNX Conv {idx:3d}: shape={w_shape}")

if __name__ == "__main__":
    check_mapping()
