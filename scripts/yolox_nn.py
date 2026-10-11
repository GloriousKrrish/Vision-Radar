import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act == "silu" else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        return self.pconv(self.dconv(x))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=True):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1)
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
        self.conv1 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1)
        self.conv2 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, ksize=1, stride=1)
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

class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, ksize=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        features = [x] + [m(x) for m in self.m]
        return self.conv2(torch.cat(features, dim=1))

class YOLOPAFPN(nn.Module):
    def __init__(self, depth=0.33, width=0.25, in_channels=[256, 512, 1024], depthwise=True):
        super().__init__()
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        # Stem
        self.stem = BaseConv(3, int(64 * width), ksize=3, stride=2) # 320x320

        # Dark2
        self.dark2 = nn.Sequential(
            Conv(int(64 * width), int(128 * width), ksize=3, stride=2), # 160x160
            CSPLayer(int(128 * width), int(128 * width), n=round(3 * depth), depthwise=depthwise),
        )

        # Dark3
        self.dark3 = nn.Sequential(
            Conv(int(128 * width), int(256 * width), ksize=3, stride=2), # 80x80 (P3)
            CSPLayer(int(256 * width), int(256 * width), n=round(9 * depth), depthwise=depthwise),
        )

        # Dark4
        self.dark4 = nn.Sequential(
            Conv(int(256 * width), int(512 * width), ksize=3, stride=2), # 40x40 (P4)
            CSPLayer(int(512 * width), int(512 * width), n=round(9 * depth), depthwise=depthwise),
        )

        # Dark5
        self.dark5 = nn.Sequential(
            Conv(int(512 * width), int(1024 * width), ksize=3, stride=2), # 20x20 (P5)
            SPPBottleneck(int(1024 * width), int(1024 * width)),
            CSPLayer(int(1024 * width), int(1024 * width), n=round(3 * depth), shortcut=False, depthwise=depthwise),
        )

        # Top-down FPN
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce_conv1 = BaseConv(int(1024 * width), int(512 * width), ksize=1, stride=1)
        self.C3p4 = CSPLayer(int(1024 * width), int(512 * width), n=round(3 * depth), shortcut=False, depthwise=depthwise)

        self.reduce_conv2 = BaseConv(int(512 * width), int(256 * width), ksize=1, stride=1)
        self.C3p3 = CSPLayer(int(512 * width), int(256 * width), n=round(3 * depth), shortcut=False, depthwise=depthwise)

        # Bottom-up PAN
        self.bu_conv2 = Conv(int(256 * width), int(256 * width), ksize=3, stride=2) # 64
        # p_bu1 channels: bu_conv2 (64) + fpn_out1 (128) = 192 -> out 128
        self.C3n3 = CSPLayer(int(192 * width) if width==1.0 else 192, int(512 * width), n=round(3 * depth), shortcut=False, depthwise=depthwise)

        self.bu_conv1 = Conv(int(512 * width), int(512 * width), ksize=3, stride=2) # 128
        # p_bu2 channels: bu_conv1 (128) + x3 (256) = 384 -> out 256
        self.C3n4 = CSPLayer(int(384 * width) if width==1.0 else 384, int(1024 * width), n=round(3 * depth), shortcut=False, depthwise=depthwise)

    def forward(self, input):
        x = self.stem(input)
        x = self.dark2(x)
        x1 = self.dark3(x)  # stride 8 (P3: 256*width -> 64)
        x2 = self.dark4(x1) # stride 16 (P4: 512*width -> 128)
        x3 = self.dark5(x2) # stride 32 (P5: 1024*width -> 256)

        # Top-down
        f_expr1 = self.reduce_conv1(x3) # 128
        f_up1 = self.upsample(f_expr1)
        f_expr1 = torch.cat([f_up1, x2], 1) # 128 + 128 = 256
        fpn_out1 = self.C3p4(f_expr1) # 128

        f_expr2 = self.reduce_conv2(fpn_out1) # 64
        f_up2 = self.upsample(f_expr2)
        f_expr2 = torch.cat([f_up2, x1], 1) # 64 + 64 = 128
        pan_out2 = self.C3p3(f_expr2) # 64 (P3_out)

        # Bottom-up
        p_bu1 = self.bu_conv2(pan_out2) # 64
        p_bu1 = torch.cat([p_bu1, fpn_out1], 1) # 64 + 128 = 192 -> 128
        pan_out1 = self.C3n3(p_bu1) # 128 (P4_out)

        p_bu2 = self.bu_conv1(pan_out1) # 128
        p_bu2 = torch.cat([p_bu2, x3], 1) # 128 + 256 = 384 -> 256
        pan_out0 = self.C3n4(p_bu2) # 256 (P5_out)

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
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            in_ch = int(in_channels[i] * width)
            out_ch = int(256 * width)
            self.stems.append(BaseConv(in_ch, out_ch, ksize=1, stride=1))
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
            self.cls_preds.append(nn.Conv2d(out_ch, self.num_classes, kernel_size=1, stride=1))
            self.reg_preds.append(nn.Conv2d(out_ch, 4, kernel_size=1, stride=1))
            self.obj_preds.append(nn.Conv2d(out_ch, 1, kernel_size=1, stride=1))

    def forward(self, xin):
        outputs = []
        for k, x in enumerate(xin):
            x = self.stems[k](x)
            cls_feat = self.cls_convs[k](x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = self.reg_convs[k](x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # Concatenate [reg (4), obj (1), cls (num_classes)] along channel dim
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            batch_size, channels, h, w = output.shape
            output = output.view(batch_size, channels, h * w).permute(0, 2, 1) # [B, HW, 5 + num_classes]
            outputs.append(output)

        return torch.cat(outputs, dim=1) # [B, 8400, 5 + num_classes]

class YOLOXNano(nn.Module):
    def __init__(self, num_classes=4, depth=0.33, width=0.25, depthwise=True):
        super().__init__()
        self.backbone = YOLOPAFPN(depth=depth, width=width, depthwise=depthwise)
        self.head = YOLOXHead(num_classes=num_classes, width=width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs = self.backbone(x)
        outputs = self.head(fpn_outs)
        return outputs

if __name__ == "__main__":
    dummy_input = torch.zeros(1, 3, 640, 640)
    model = YOLOXNano(num_classes=4)
    out = model(dummy_input)
    print("YOLOXNano output shape:", out.shape)
