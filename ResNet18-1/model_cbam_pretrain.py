"""ResNet18 + CBAM，使用 ImageNet 预训练权重初始化

结构命名与 torchvision 官方 resnet18 完全一致(conv1/bn1/layer1-4/fc)，
因此可直接 load_state_dict 加载官方预训练权重；
仅在每个 BasicBlock 残差相加前插入 CBAM，并替换分类头为 2 类。
"""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from attention import CBAM


class BasicBlockCBAM(nn.Module):
    """torchvision BasicBlock 同构 + CBAM"""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        # CBAM 注意力：通道 + 空间
        self.cbam = CBAM(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)  # 注意力加权后与恒等映射相加

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18CBAMPretrain(nn.Module):
    """与 torchvision resnet18 相同拓扑，layer 使用 BasicBlockCBAM"""

    def __init__(self, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlockCBAM(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlockCBAM(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_pretrained_model():
    """加载 ImageNet 预训练权重并替换为 2 类分类头"""
    model = ResNet18CBAMPretrain(num_classes=1000)
    weights = ResNet18_Weights.IMAGENET1K_V1
    state = resnet18(weights=weights).state_dict()
    # 仅加载结构匹配的预训练层(conv/bn/layer)，CBAM 新参数与 fc 保持随机初始化
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"预训练加载: 匹配层已加载, 缺失新参数 {len(missing)} 个(CBAM/fc), 无关参数 {len(unexpected)} 个(fc)")
    model.fc = nn.Linear(512, 2)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pretrained_model().to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    print("输出:", model(x).shape)
    print("参数量:", sum(p.numel() for p in model.parameters()))