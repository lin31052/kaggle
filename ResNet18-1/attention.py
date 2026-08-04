"""注意力机制模块：SE 与 CBAM

- SE   : Squeeze-and-Excitation，仅通道注意力
- CBAM : Convolutional Block Attention Module，通道注意力 + 空间注意力
"""
import torch
from torch import nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力

    - Squeeze   : 全局平均池化，把每个通道压缩成 1 个标量
    - Excitation: 两个全连接层学习通道权重，Sigmoid 归一化到 (0, 1)
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (N, C, H, W) -> 压缩为 (N, C) -> 学习通道权重 -> 还原为 (N, C, 1, 1)
        w = self.squeeze(x).view(x.size(0), -1)
        w = self.excitation(w).view(x.size(0), x.size(1), 1, 1)
        return x * w


class ChannelAttention(nn.Module):
    """CBAM 通道注意力：平均池化 + 最大池化双分支，共享 MLP 后相加"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化与最大池化分别描述整体响应与显著响应
        avg = torch.mean(x, dim=(2, 3))                 # (N, C)
        mx = torch.max(x, dim=3)[0].max(dim=2)[0]       # (N, C)
        w = self.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * w.view(x.size(0), x.size(1), 1, 1)


class SpatialAttention(nn.Module):
    """CBAM 空间注意力：通道维拼接 avg/max 后，用 7x7 卷积学习空间权重"""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)        # (N, 1, H, W)
        mx = torch.max(x, dim=1, keepdim=True)[0]       # (N, 1, H, W)
        fused = torch.cat([avg, mx], dim=1)             # (N, 2, H, W)
        w = self.sigmoid(self.conv(fused))
        return x * w


class CBAM(nn.Module):
    """CBAM：先通道注意力，再空间注意力，串联构成"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x