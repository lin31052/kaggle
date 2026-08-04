"""ResNet18 + SE 注意力（Squeeze-and-Excitation）

与 model.py 的 ResNet18 结构完全一致，仅在每个残差块内、
残差分支相加之前插入 SEBlock 通道注意力。
"""
import torch
from torch import nn
from attention import SEBlock


class ResidualSE(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        # SE 注意力：作用于残差分支的输出特征图
        self.se = SEBlock(num_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.se(y)  # 通道注意力加权后，再与恒等映射相加
        if self.conv3:
            x = self.conv3(x)
        y = self.ReLU(y + x)
        return y


class ResNet18SE(nn.Module):
    def __init__(self, residual_block):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(residual_block(64, 64, use_1conv=False, strides=1),
                                residual_block(64, 64, use_1conv=False, strides=1))

        self.b3 = nn.Sequential(residual_block(64, 128, use_1conv=True, strides=2),
                                residual_block(128, 128, use_1conv=False, strides=1))

        self.b4 = nn.Sequential(residual_block(128, 256, use_1conv=True, strides=2),
                                residual_block(256, 256, use_1conv=False, strides=1))

        self.b5 = nn.Sequential(residual_block(256, 512, use_1conv=True, strides=2),
                                residual_block(512, 512, use_1conv=False, strides=1))

        self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(512, 2))

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18SE(ResidualSE).to(device)
    print(model)
    print("参数量:", sum(p.numel() for p in model.parameters()))