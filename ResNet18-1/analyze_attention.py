"""量化分析 Grad-CAM / CBAM 空间注意力热力图分布

对错误样本与正确对照样本，计算：
- 高响应区质心位置（是否居中于口罩区域）
- 中心区域响应占比（中心 50% 面积内的高响应比例）
- 高响应连通区域数量（是否碎片化/被背景吸引）
"""
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np

from model_cbam_pretrain import build_pretrained_model

NORMALIZE = transforms.Normalize(
    [0.17263485, 0.15147247, 0.14267451],
    [0.0736155, 0.06216329, 0.05930814],
)
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORMALIZE])

SAMPLES = [
    ("mask136", "data/test/mask/mask136.jpg", "错误"),
    ("mask178", "data/test/mask/mask178.jpg", "错误"),
    ("mask112", "data/test/mask/mask112.jpg", "正确"),
    ("no_mask12", "data/test/no_mask/no_mask12.jpg", "正确"),
]


class GradCAM:
    def __init__(self, model, target_layer):
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
        out[0, class_idx].backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = nn.functional.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()


def analyze(name, cam, label):
    """统计热力图分布指标"""
    h, w = cam.shape
    # 高响应掩码（前 30% 分位）
    thr = np.quantile(cam, 0.70)
    mask = cam >= thr
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask)
    cy, cx = ys.mean() / h, xs.mean() / w  # 质心（归一化 0~1）
    # 中心区域（水平/垂直均取 25%~75%）内高响应占比
    center_mask = (np.abs(ys - h / 2) < h / 4) & (np.abs(xs - w / 2) < w / 4)
    center_ratio = center_mask.sum() / mask.sum()
    # 高响应区占全图面积比
    area_ratio = mask.sum() / (h * w)
    print(f"[{name}] ({label}) 质心=({cx:.2f},{cy:.2f}) 中心占比={center_ratio:.2f} 面积占比={area_ratio:.2f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_pretrained_model().to(device).eval()
    model.load_state_dict(torch.load("best_model_cbam_pt.pth", map_location=device))
    cam_gt = GradCAM(model, model.layer4[-1].conv2)
    cam_gt.model = model

    for name, path, label in SAMPLES:
        img = Image.open(path).convert("RGB")
        x = TF(img).unsqueeze(0).to(device)
        out = model(x)
        pred = out.argmax(dim=1).item()
        true = 0 if "no_mask" not in path else 1
        print(f"--- {name}: 真实={'戴口罩' if true==0 else '不带口罩'} 预测={'戴口罩' if pred==0 else '不带口罩'}")
        cam = cam_gt(x, true)
        analyze(f"{name}/GradCAM(真实类)", cam, label)
        cam2 = cam_gt(x, pred)
        analyze(f"{name}/GradCAM(预测类)", cam2, label)
    cam_gt.remove()


if __name__ == "__main__":
    main()