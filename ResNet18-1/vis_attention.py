"""注意力可视化：Grad-CAM + CBAM 空间注意力

针对测试集错误样本(mask10/mask108)与对照样本，输出四联图：
  原图 | 真实类 Grad-CAM | 预测类 Grad-CAM | CBAM 空间注意力
用于判断模型是否被口罩以外的背景区域干扰。

用法:
  python vis_attention.py
  python vis_attention.py --images data/test/mask/mask10.jpg data/test/mask/mask112.jpg
"""
import argparse
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_cbam_pretrain import build_pretrained_model

# 中文字体（Windows 下 SimHei 可用），避免标题乱码
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 与训练端一致：本数据集统一使用数据集自身统计量归一化
NORMALIZE = transforms.Normalize(
    [0.17263485, 0.15147247, 0.14267451],
    [0.0736155, 0.06216329, 0.05930814],
)
CLASSES = ["戴口罩", "不带口罩"]


def load_image(path):
    img = Image.open(path).convert("RGB")
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORMALIZE])
    x = tf(img).unsqueeze(0)
    return img, x


class GradCAM:
    """针对目标卷积层的最小 Grad-CAM 实现（register_hook 方式，无外部依赖）"""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        out = self.model(x)
        score = out[0, class_idx]
        score.backward(retain_graph=True)
        # 梯度全局平均池化作为通道权重
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1,1,H,W)
        cam = nn.functional.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return out, cam

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()


def cbam_spatial_attention(model, x):
    """取 layer4 最后一个 block 的 CBAM 空间注意力权重"""
    sa = model.layer4[-1].cbam.sa
    model.zero_grad()
    with torch.no_grad():
        out = model(x)
        # 手动重放得到 sa 权重：直接从 block 输出反推不可行，改为前向到 sa
        # 重放 layer4[-1] 前的特征
        def forward_to_sa(model, x):
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4[0](x)
            return model.layer4[1](x)  # 最后一个 block 输出（含 cbam 加权）

        feat = forward_to_sa(model, x)
        # 手动重放 cbam.sa 的 sigmoid 权重
        avg = feat.mean(dim=1, keepdim=True)
        mx = feat.max(dim=1, keepdim=True)[0]
        fused = torch.cat([avg, mx], dim=1)
        w = sa.sigmoid(sa.conv(fused))
        w = nn.functional.interpolate(w, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out, w.squeeze().cpu().numpy()


def overlay(img_pil, heat, alpha=0.5):
    """把热力图叠加到原图(224x224)"""
    img = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0
    cmap = plt.get_cmap("jet")(heat)[..., :3]
    blended = (1 - alpha) * img + alpha * cmap
    return np.clip(blended, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", default=[
        "data/test/mask/mask136.jpg",   # 错误样本1：戴口罩 -> 不带口罩
        "data/test/mask/mask178.jpg",   # 错误样本2：戴口罩 -> 不带口罩
        "data/test/mask/mask112.jpg",   # 对照：正确识别戴口罩
        "data/test/no_mask/no_mask12.jpg",  # 对照：正确识别不带口罩
    ])
    parser.add_argument("--weight", default="best_model_cbam_pt.pth")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_pretrained_model().to(device).eval()
    model.load_state_dict(torch.load(args.weight, map_location=device))

    # Grad-CAM 目标：layer4 最后一个 block 的 conv2（含 512 通道，7x7 特征图）
    cam_gt = GradCAM(model, model.layer4[-1].conv2)

    for i, path in enumerate(args.images):
        img_pil, x = load_image(path)
        x = x.to(device)
        out = model(x)
        prob = torch.softmax(out, dim=1)
        pred = out.argmax(dim=1).item()
        # 从路径推断真实标签（注意 no_mask 也含 "mask" 子串，先判断 no_mask）
        true = None
        if "no_mask" in path:
            true = 1
        elif "mask" in path:
            true = 0

        # 预测类 Grad-CAM
        _, cam_pred = cam_gt(x, pred)
        # 真实类 Grad-CAM（若不同则展示模型本应关注的区域）
        _, cam_true = cam_gt(x, true)

        # CBAM 空间注意力（layer4 末块）
        _, sa = cbam_spatial_attention(model, x)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_pil.resize((224, 224)))
        axes[0].set_title(f"原图\n真实={CLASSES[true]}")
        axes[1].imshow(overlay(img_pil, cam_true))
        axes[1].set_title("真实类 Grad-CAM")
        axes[2].imshow(overlay(img_pil, cam_pred))
        axes[2].set_title(f"预测类 Grad-CAM\n预测={CLASSES[pred]} ({prob[0,pred].item():.2f})")
        axes[3].imshow(overlay(img_pil, sa))
        axes[3].set_title("CBAM 空间注意力(layer4)")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        out_png = f"vis_attn_{i}.png"
        plt.savefig(out_png, dpi=120)
        plt.close()
        print(f"[{path}] 真实={CLASSES[true]} 预测={CLASSES[pred]} 置信度={prob[0,pred].item():.3f} -> {out_png}")

    cam_gt.remove()


if __name__ == "__main__":
    main()