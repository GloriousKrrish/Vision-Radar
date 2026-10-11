import os
import sys
import glob
import time
import json
import math
import hashlib
import random
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "packages"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import onnx
import onnxruntime as ort

from visionradar.eval.schemas import GroundTruthDetection
from visionradar.eval.dataset_adapters import DetectionDatasetAdapter
from visionradar.eval.evaluator import BaselineEvaluator, VehicleClass

# Ensure deterministic seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- Configuration & Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(
    BASE_DIR, "DATASETS", "vehicle-detection-project-main", "vehicle-detection-project-main"
)
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATASET_ROOT, "train", "labels")
VAL_IMG_DIR = os.path.join(DATASET_ROOT, "val", "images")
VAL_LBL_DIR = os.path.join(DATASET_ROOT, "val", "labels")
TEST_IMG_DIR = os.path.join(DATASET_ROOT, "test", "images")
TEST_LBL_DIR = os.path.join(DATASET_ROOT, "test", "labels")

PRETRAINED_ONNX_PATH = os.path.join(BASE_DIR, "data", "models", "yolox_nano.onnx")
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, "models", "experiments", "phase5_2")
OUTPUT_DEBUG_DIR = os.path.join(BASE_DIR, "data", "debug", "phase5_2")
OUTPUT_EXP_DIR = os.path.join(BASE_DIR, "data", "experiments")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "PHASE_5_2_FINE_TUNING_REPORT.md")

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DEBUG_DIR, exist_ok=True)
os.makedirs(OUTPUT_EXP_DIR, exist_ok=True)

VALID_CLASSES = {0: "bus", 1: "car", 2: "motorcycle", 3: "truck"}

# --- Dataset Loader ---
class RoboflowYOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, input_size=(416, 416), augment=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.augment = augment

        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.*")))
        self.img_basenames = [os.path.splitext(os.path.basename(f))[0] for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base = self.img_basenames[idx]
        lbl_path = os.path.join(self.label_dir, f"{base}.txt")

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            img_resized = img.resize(self.input_size, Image.BILINEAR)
            img_np = np.array(img_resized, dtype=np.float32) / 255.0

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        boxes = []
        labels = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        boxes.append([cx, cy, w, h])
                        labels.append(cls_id)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        if self.augment and random.random() > 0.5:
            img_tensor = torch.flip(img_tensor, dims=[2])
            if len(boxes) > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]

        return img_tensor, boxes, labels, base, (orig_w, orig_h)

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    boxes = [b[1] for b in batch]
    labels = [b[2] for b in batch]
    bases = [b[3] for b in batch]
    orig_sizes = [b[4] for b in batch]
    return imgs, boxes, labels, bases, orig_sizes

# --- PyTorch YOLOX-Nano Architecture ---
class FusedConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, groups=1, act="silu"):
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

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=True):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = FusedDWConv if depthwise else FusedConv
        self.conv1 = FusedConv(in_channels, hidden_channels, ksize=1, stride=1)
        self.conv2 = Conv(hidden_channels, out_channels, ksize=3, stride=1)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=True):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = FusedConv(in_channels, hidden_channels, ksize=1, stride=1)
        self.conv2 = FusedConv(in_channels, hidden_channels, ksize=1, stride=1)
        self.conv3 = FusedConv(2 * hidden_channels, out_channels, ksize=1, stride=1)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise) for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x_all = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x_all)

class CSPLayerPAN(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, n=1, shortcut=False, expansion=0.5, depthwise=True):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = FusedConv(in_channels1, hidden_channels, ksize=1, stride=1)
        self.conv2 = FusedConv(in_channels2, hidden_channels, ksize=1, stride=1)
        self.conv3 = FusedConv(2 * hidden_channels, out_channels, ksize=1, stride=1)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise) for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x1, x2):
        x_1 = self.conv1(x1)
        x_2 = self.conv2(x2)
        x_1 = self.m(x_1)
        x_all = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x_all)

class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = FusedConv(in_channels, hidden_channels, ksize=1, stride=1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = FusedConv(conv2_channels, out_channels, ksize=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        features = [x] + [m(x) for m in self.m]
        return self.conv2(torch.cat(features, dim=1))

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

class YOLOPAFPN(nn.Module):
    def __init__(self, depth=0.33, width=0.25, depthwise=True):
        super().__init__()
        Conv = FusedDWConv if depthwise else FusedConv

        self.stem = Focus(3, int(64 * width), ksize=3, stride=1)

        self.dark2 = nn.Sequential(
            Conv(int(64 * width), int(128 * width), ksize=3, stride=2),
            CSPLayer(int(128 * width), int(128 * width), n=round(3 * depth), depthwise=depthwise),
        )

        self.dark3 = nn.Sequential(
            Conv(int(128 * width), int(256 * width), ksize=3, stride=2),
            CSPLayer(int(256 * width), int(256 * width), n=round(9 * depth), depthwise=depthwise),
        )

        self.dark4 = nn.Sequential(
            Conv(int(256 * width), int(512 * width), ksize=3, stride=2),
            CSPLayer(int(512 * width), int(512 * width), n=round(9 * depth), depthwise=depthwise),
        )

        self.dark5 = nn.Sequential(
            Conv(int(512 * width), int(1024 * width), ksize=3, stride=2),
            SPPBottleneck(int(1024 * width), int(1024 * width)),
            CSPLayer(int(1024 * width), int(1024 * width), n=round(3 * depth), shortcut=False, depthwise=depthwise),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce_conv1 = FusedConv(int(1024 * width), int(512 * width), ksize=1, stride=1)
        self.C3p4 = CSPLayer(256, 128, n=round(3 * depth), shortcut=False, depthwise=depthwise)

        self.reduce_conv2 = FusedConv(int(512 * width), int(256 * width), ksize=1, stride=1)
        self.C3p3 = CSPLayer(128, 64, n=round(3 * depth), shortcut=False, depthwise=depthwise)

        self.bu_conv2 = Conv(int(256 * width), int(256 * width), ksize=3, stride=2)
        self.C3n3 = CSPLayer(192, 128, n=round(3 * depth), shortcut=False, depthwise=depthwise)

        self.bu_conv1 = Conv(int(512 * width), int(512 * width), ksize=3, stride=2)
        self.C3n4 = CSPLayer(384, 256, n=round(3 * depth), shortcut=False, depthwise=depthwise)

    def forward(self, input):
        x = self.stem(input)
        x = self.dark2(x)
        x1 = self.dark3(x)
        x2 = self.dark4(x1)
        x3 = self.dark5(x2)

        f_expr1 = self.reduce_conv1(x3)
        f_up1 = self.upsample(f_expr1)
        f_expr1 = torch.cat([f_up1, x2], 1)
        fpn_out1 = self.C3p4(f_expr1)

        f_expr2 = self.reduce_conv2(fpn_out1)
        f_up2 = self.upsample(f_expr2)
        f_expr2 = torch.cat([f_up2, x1], 1)
        pan_out2 = self.C3p3(f_expr2)

        p_bu1 = self.bu_conv2(pan_out2)
        p_bu1 = torch.cat([p_bu1, fpn_out1], 1)
        pan_out1 = self.C3n3(p_bu1)

        p_bu2 = self.bu_conv1(pan_out1)
        p_bu2 = torch.cat([p_bu2, x3], 1)
        pan_out0 = self.C3n4(p_bu2)

        return (pan_out2, pan_out1, pan_out0)

class YOLOXHead(nn.Module):
    def __init__(self, num_classes=4, width=0.25, in_channels=[256, 512, 1024], depthwise=True):
        super().__init__()
        self.num_classes = num_classes
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = FusedDWConv if depthwise else FusedConv

        for i in range(len(in_channels)):
            in_ch = int(in_channels[i] * width)
            out_ch = int(256 * width)
            self.stems.append(FusedConv(in_ch, out_ch, ksize=1, stride=1))
            self.cls_convs.append(
                nn.Sequential(
                    Conv(out_ch, out_ch, ksize=3, stride=1),
                    Conv(out_ch, out_ch, ksize=3, stride=1),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    Conv(out_ch, out_ch, ksize=3, stride=1),
                    Conv(out_ch, out_ch, ksize=3, stride=1),
                )
            )
            self.cls_preds.append(nn.Conv2d(out_ch, self.num_classes, kernel_size=1, stride=1, bias=True))
            self.reg_preds.append(nn.Conv2d(out_ch, 4, kernel_size=1, stride=1, bias=True))
            self.obj_preds.append(nn.Conv2d(out_ch, 1, kernel_size=1, stride=1, bias=True))

    def forward(self, xin):
        outputs = []
        for k, x in enumerate(xin):
            x = self.stems[k](x)
            cls_feat = self.cls_convs[k](x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = self.reg_convs[k](x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            batch_size, channels, h, w = output.shape
            output = output.view(batch_size, channels, h * w).permute(0, 2, 1)
            outputs.append(output)

        return torch.cat(outputs, dim=1)

class YOLOXNanoTorch(nn.Module):
    def __init__(self, num_classes=4, depth=0.33, width=0.25, depthwise=True):
        super().__init__()
        self.backbone = YOLOPAFPN(depth=depth, width=width, depthwise=depthwise)
        self.head = YOLOXHead(num_classes=num_classes, width=width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs = self.backbone(x)
        outputs = self.head(fpn_outs)
        return outputs

def generate_yolox_grids(input_size=(416, 416)):
    strides = [8, 16, 32]
    grids = []
    stride_tensor = []

    for s in strides:
        h, w = input_size[1] // s, input_size[0] // s
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        grid = torch.stack((x, y), dim=-1).view(-1, 2).float()
        grids.append(grid)
        stride_tensor.append(torch.full((h * w, 1), float(s)))

    grids = torch.cat(grids, dim=0) # [3549, 2]
    strides = torch.cat(stride_tensor, dim=0) # [3549, 1]
    return grids, strides

def load_pretrained_onnx_weights(pytorch_model, onnx_path=PRETRAINED_ONNX_PATH):
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    initializers = {w.name: onnx.numpy_helper.to_array(w) for w in graph.initializer}
    conv_nodes = [n for n in graph.node if n.op_type == 'Conv']

    onnx_weights = []
    for n in conv_nodes:
        w_arr = initializers[n.input[1]].copy()
        b_arr = initializers[n.input[2]].copy() if len(n.input) > 2 and n.input[2] in initializers else None
        onnx_weights.append((w_arr, b_arr, n.name))

    py_convs = [m for m in pytorch_model.modules() if isinstance(m, nn.Conv2d)]
    
    shape_pool = {}
    for w_arr, b_arr, name in onnx_weights:
        s = tuple(w_arr.shape)
        if s not in shape_pool:
            shape_pool[s] = []
        shape_pool[s].append((w_arr, b_arr, name))

    coco_indices = [5, 2, 3, 7] # bus, car, motorcycle, truck
    loaded = 0

    with torch.no_grad():
        for idx, p_module in enumerate(py_convs):
            target_shape = tuple(p_module.weight.shape)
            
            if target_shape[0] == 4 and target_shape[1] == 64 and target_shape[2] == 1 and target_shape[3] == 1:
                if (80, 64, 1, 1) in shape_pool and len(shape_pool[(80, 64, 1, 1)]) > 0:
                    w_arr, b_arr, name = shape_pool[(80, 64, 1, 1)].pop(0)
                    w_arr = w_arr[coco_indices]
                    if b_arr is not None:
                        b_arr = b_arr[coco_indices]
                    p_module.weight.copy_(torch.from_numpy(w_arr))
                    if p_module.bias is not None and b_arr is not None:
                        p_module.bias.copy_(torch.from_numpy(b_arr))
                    loaded += 1
                    continue

            if target_shape in shape_pool and len(shape_pool[target_shape]) > 0:
                w_arr, b_arr, name = shape_pool[target_shape].pop(0)
                p_module.weight.copy_(torch.from_numpy(w_arr))
                if p_module.bias is not None and b_arr is not None:
                    p_module.bias.copy_(torch.from_numpy(b_arr))
                loaded += 1

    print(f"Pretrained ONNX weights successfully loaded: {loaded} / {len(py_convs)} modules matched.")
    return pytorch_model

# --- Target Matching & Loss Function ---
def compute_yolox_loss(pred_outputs, batch_boxes, batch_labels, grids, strides):
    # pred_outputs: [B, 3549, 9] (reg_x, reg_y, reg_w, reg_h, obj, cls_0..3)
    batch_size = pred_outputs.shape[0]
    num_anchors = pred_outputs.shape[1]
    device = pred_outputs.device

    reg_preds = pred_outputs[:, :, :4]
    obj_preds = pred_outputs[:, :, 4]
    cls_preds = pred_outputs[:, :, 5:]

    grids = grids.to(device)
    strides = strides.to(device)

    # Target tensors
    target_obj = torch.zeros((batch_size, num_anchors), dtype=torch.float32, device=device)
    target_cls = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)
    target_reg = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)
    pos_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=device)

    for b in range(batch_size):
        gt_boxes = batch_boxes[b] # [N, 4] normalized cx, cy, w, h
        gt_labels = batch_labels[b] # [N]
        if len(gt_boxes) == 0:
            continue

        gt_cx = gt_boxes[:, 0] * 416.0
        gt_cy = gt_boxes[:, 1] * 416.0
        gt_w = gt_boxes[:, 2] * 416.0
        gt_h = gt_boxes[:, 3] * 416.0

        for i in range(len(gt_boxes)):
            cls_id = gt_labels[i].item()
            c_x, c_y, w, h = gt_cx[i].item(), gt_cy[i].item(), gt_w[i].item(), gt_h[i].item()

            # Center grid assignment
            grid_x_idx = grids[:, 0]
            grid_y_idx = grids[:, 1]
            anchor_cx = (grid_x_idx + 0.5) * strides[:, 0]
            anchor_cy = (grid_y_idx + 0.5) * strides[:, 0]

            # In box condition
            in_box = (anchor_cx >= (c_x - w / 2.0)) & (anchor_cx <= (c_x + w / 2.0)) & \
                     (anchor_cy >= (c_y - h / 2.0)) & (anchor_cy <= (c_y + h / 2.0))

            pos_indices = torch.where(in_box)[0]
            if len(pos_indices) == 0:
                # Closest anchor fallback
                dist = (anchor_cx - c_x)**2 + (anchor_cy - c_y)**2
                pos_indices = torch.tensor([torch.argmin(dist).item()], device=device)

            pos_mask[b, pos_indices] = True
            target_obj[b, pos_indices] = 1.0
            target_cls[b, pos_indices, cls_id] = 1.0

            # Target regression offsets: (c_x / stride) - grid_x, (c_y / stride) - grid_y, ln(w / stride), ln(h / stride)
            tx = (c_x / strides[pos_indices, 0]) - grids[pos_indices, 0]
            ty = (c_y / strides[pos_indices, 0]) - grids[pos_indices, 1]
            tw = torch.log(torch.clamp(torch.tensor(w, device=device) / strides[pos_indices, 0], min=1e-6))
            th = torch.log(torch.clamp(torch.tensor(h, device=device) / strides[pos_indices, 0], min=1e-6))
            target_reg[b, pos_indices] = torch.stack([tx, ty, tw, th], dim=-1)

    # Losses
    bce_obj = nn.BCEWithLogitsLoss(reduction="mean")
    bce_cls = nn.BCEWithLogitsLoss(reduction="sum")
    l1_reg = nn.SmoothL1Loss(reduction="sum")

    loss_obj = bce_obj(obj_preds, target_obj)

    num_pos = torch.sum(pos_mask).item()
    if num_pos > 0:
        loss_cls = bce_cls(cls_preds[pos_mask], target_cls[pos_mask]) / max(1.0, num_pos)
        loss_reg = l1_reg(reg_preds[pos_mask], target_reg[pos_mask]) / max(1.0, num_pos)
    else:
        loss_cls = torch.tensor(0.0, device=device)
        loss_reg = torch.tensor(0.0, device=device)

    total_loss = 1.0 * loss_obj + 1.0 * loss_cls + 2.0 * loss_reg
    return total_loss, loss_obj.item(), loss_cls.item(), loss_reg.item(), num_pos

def decode_yolox_predictions(pred_tensor, grids, strides, input_size=(416, 416)):
    box_reg = pred_tensor[:, :, :4]
    obj_log = pred_tensor[:, :, 4:5]
    cls_log = pred_tensor[:, :, 5:]

    device = pred_tensor.device
    grids = grids.to(device)
    strides = strides.to(device)

    pred_cx = (box_reg[:, :, 0] + grids[:, 0]) * strides[:, 0]
    pred_cy = (box_reg[:, :, 1] + grids[:, 1]) * strides[:, 0]
    pred_w = torch.exp(box_reg[:, :, 2]) * strides[:, 0]
    pred_h = torch.exp(box_reg[:, :, 3]) * strides[:, 0]

    x1 = pred_cx - pred_w / 2.0
    y1 = pred_cy - pred_h / 2.0
    x2 = pred_cx + pred_w / 2.0
    y2 = pred_cy + pred_h / 2.0

    boxes_decoded = torch.stack([x1, y1, x2, y2], dim=-1)
    obj_scores = torch.sigmoid(obj_log)
    cls_scores = torch.sigmoid(cls_log)

    return boxes_decoded, obj_scores, cls_scores

print("Loss computation & decoding functions ready.")
