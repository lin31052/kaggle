"""定位独立测试集中的错误样本（保存文件名+真实/预测标签+置信度）"""
import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
import argparse

# 与 model_test 相同的注册表
MODEL_REGISTRY = {
    "cbam_pt": ("model_cbam_pretrain", "build_pretrained_model", None, "factory"),
}

def build_model(model_name):
    import importlib
    module_name, net_cls, block_cls, build_mode = MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_name)
    if build_mode == "factory":
        return getattr(module, net_cls)()
    return getattr(module, net_cls)(getattr(module, block_cls))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cbam_pt")
    parser.add_argument("--weight", default=None, help="权重路径，默认 best_model_<model>.pth")
    args = parser.parse_args()

    model = build_model(args.model)
    weight = args.weight or f"best_model_{args.model}.pth"
    model.load_state_dict(torch.load(weight, map_location="cpu"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # 数据集自身统计量（与训练一致）
    normalize = transforms.Normalize([0.17263485, 0.15147247, 0.14267451],
                                     [0.0736155, 0.06216329, 0.05930814])
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    test_data = ImageFolder(r"data\test", transform=test_transform)
    loader = Data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    classes = ['戴口罩', '不带口罩']
    wrong = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            out = model(x)
            prob = torch.softmax(out, dim=1)
            pred = out.argmax(dim=1).item()
            if pred != y.item():
                # 用真实样本索引 idx 取路径，而不是错误计数 len(wrong)
                wrong.append((test_data.samples[idx], y.item(), pred, prob[0, y.item()].item()))

    print(f"错误样本数: {len(wrong)} / {len(test_data)}")
    for path, true_lab, pred_lab, conf in wrong:
        print(f"文件: {path}")
        print(f"  真实={classes[true_lab]} 预测={classes[pred_lab]} 真实类别置信度={conf:.4f}")

if __name__ == "__main__":
    main()