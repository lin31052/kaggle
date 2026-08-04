import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.datasets import ImageFolder
from PIL import Image
import argparse

# 模型注册表：model 参数 -> (模型模块, 网络类, 残差块类, 构建方式)
# 带 _aug 后缀的条目仅用于加载对应增强训练权重，网络结构相同
# 构建方式: "block" = net_cls(block_cls), "factory" = 模块内的 build 函数
MODEL_REGISTRY = {
    "baseline": ("model", "ResNet18", "Residual", "block"),
    "se": ("model_se", "ResNet18SE", "ResidualSE", "block"),
    "cbam": ("model_cbam", "ResNet18CBAM", "ResidualCBAM", "block"),
    "cbam_aug": ("model_cbam", "ResNet18CBAM", "ResidualCBAM", "block"),
    "cbam_pt": ("model_cbam_pretrain", "build_pretrained_model", None, "factory"),
}

def build_model(model_name):
    """根据名称动态导入并构造模型"""
    import importlib
    module_name, net_cls, block_cls, build_mode = MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_name)
    if build_mode == "factory":
        # 预训练模型走工厂函数（内部加载 ImageNet 权重）
        builder = getattr(module, net_cls)
        return builder()
    net_cls = getattr(module, net_cls)
    block_cls = getattr(module, block_cls)
    return net_cls(block_cls)

def test_data_process(pt_norm=False):
    # 定义数据集的路径
    ROOT_TRAIN = 'data/test'

    # 预训练模型使用 ImageNet 统计量归一化（与训练端一致）
    if pt_norm:
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize([0.17263485, 0.15147247, 0.14267451], [0.0736155,  0.06216329, 0.05930814])
    # 定义数据集处理方法变量
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    # 加载数据集
    test_data = ImageFolder(ROOT_TRAIN, transform=test_transform)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)
    return test_dataloader


def test_model_process(model, test_dataloader):
    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # 讲模型放入到训练设备中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将特征放入到测试设备中
            test_data_x = test_data_x.to(device)
            # 将标签放入到测试设备中
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为测试数据集，输出为对每个样本的预测值
            output= model(test_data_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则准确度test_corrects加1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本进行累加
            test_num += test_data_x.size(0)

    # 计算测试准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet18 口罩识别测试")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="baseline",
                        help="模型类型: baseline / se / cbam")
    args = parser.parse_args()

    # 加载模型
    model = build_model(args.model)
    model.load_state_dict(torch.load(f'best_model_{args.model}.pth'))
    # 与训练端一致：本数据集统一使用数据集自身统计量归一化
    # # 利用现有的模型进行模型的测试
    test_dataloader = test_data_process(pt_norm=False)
    test_model_process(model, test_dataloader)

    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    classes = ['戴口罩', '不带口罩']
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 设置模型为验证模型
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = b_y.item()
            print("预测值：",  classes[result], "------", "真实值：", classes[label])


    image = Image.open('no_mask.jfif')

    # 单图预测与数据集评估使用相同归一化（统一数据集自身统计量）
    normalize = transforms.Normalize([0.17263485, 0.15147247, 0.14267451], [0.0736155,  0.06216329, 0.05930814])
    # 定义数据集处理方法变量
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    image = test_transform(image)

    # 添加批次维度
    image = image.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        image = image.to(device)
        output = model(image)
        pre_lab = torch.argmax(output, dim=1)
        result = pre_lab.item()
    print("预测值：",  classes[result])



