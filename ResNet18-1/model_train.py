import copy
import time
import argparse

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 服务器/后台环境下禁止弹窗
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd

# 模型注册表：model 参数 -> (模型模块, 网络类, 残差块类, 构建方式)
# 构建方式: "block" = net_cls(block_cls), "factory" = 模块内的 build 函数
MODEL_REGISTRY = {
    "baseline": ("model", "ResNet18", "Residual", "block"),
    "se": ("model_se", "ResNet18SE", "ResidualSE", "block"),
    "cbam": ("model_cbam", "ResNet18CBAM", "ResidualCBAM", "block"),
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

def train_val_data_process(seed=42, aug=False, pt_norm=False, batch_size=32):
    # 固定随机种子，保证三种模型使用完全相同的 train/val 划分（公平对比）
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 定义数据集的路径
    ROOT_TRAIN = 'data/train'

    # 归一化统计量选择（实验结论：本口罩数据集用数据集自身统计量更优，
    # ImageNet 统计量在预训练微调上反而低 ~2pp，故默认关闭 --pt_norm）
    if pt_norm:
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize([0.17263485, 0.15147247, 0.14267451], [0.0736155,  0.06216329, 0.05930814])

    # 训练集变换：--aug 时叠加数据增强（随机翻转/旋转/颜色抖动），否则与第一轮完全一致
    base_transform = [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    if aug:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose(base_transform)

    # 加载数据集
    train_data = ImageFolder(ROOT_TRAIN, transform=train_transform)

    train_data, val_data = Data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=2)

    # 验证集不使用数据增强，保证与第一轮评估口径一致
    val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    val_data.dataset.transform = val_transform
    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=2)

    return train_dataloader, val_dataloader



def train_model_process(model, train_dataloader, val_dataloader, num_epochs, model_name="baseline", lr=0.001, use_cosine=False):
    # 设定训练所用到的设备，有GPU用GPU没有GPU用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器，学习率可配置（预训练微调需较小学习率）
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    # 余弦退火调度器：学习率从 lr 余弦衰减到 0，利于收敛
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) if use_cosine else None
    # 将模型放入到训练设备中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)

        # 初始化参数
        # 训练集损失函数
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 验证集损失函数
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征放入到训练设备中
            b_x = b_x.to(device)
            # 将标签放入到训练设备中
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()

            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征放入到验证设备中
            b_x = b_x.to(device)
            # 将标签放入到验证设备中
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)


            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于验证的样本数量
            val_num += b_x.size(0)

        # 计算并保存每一次迭代的loss值和准确率
        # 计算并保存训练集的loss值
        train_loss_all.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num)

        # 计算并保存验证集的loss值
        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.double().item() / val_num)

        # 余弦退火：每个 epoch 结束后调整学习率
        if scheduler is not None:
            scheduler.step()

        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前最高准确度的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use//60, time_use%60))

    # 选择最优参数，保存最优参数的模型
    model.load_state_dict(best_model_wts)
    # 按模型名区分保存，避免互相覆盖
    torch.save(best_model_wts, f"best_model_{model_name}.pth")
    print(f"最佳验证准确率: {best_acc:.4f}，已保存 best_model_{model_name}.pth")


    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                       "train_loss_all":train_loss_all,
                                       "val_loss_all":val_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_acc_all":val_acc_all,})

    return train_process


def matplot_acc_loss(train_process, save_path):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"训练曲线已保存: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ResNet18 口罩识别训练（支持 baseline/SE/CBAM/预训练）")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="baseline",
                        help="模型类型: baseline / se / cbam / cbam_pt")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（保证数据划分一致）")
    parser.add_argument("--aug", action="store_true", help="训练集启用数据增强（随机翻转/旋转/颜色抖动）")
    parser.add_argument("--lr", type=float, default=0.001, help="初始学习率（预训练微调建议 1e-4~1e-3）")
    parser.add_argument("--batch_size", type=int, default=32, help="训练/验证 batch size")
    parser.add_argument("--cosine", action="store_true", help="使用余弦退火学习率调度")
    args = parser.parse_args()

    # 带增强时用独立命名，避免覆盖同模型的无增强权重
    run_tag = f"{args.model}_aug" if args.aug else args.model

    # 加载需要的模型
    model = build_model(args.model)
    # 预训练模型也使用数据集自身归一化（实证：本数据集 ImageNet 统计量反而更低）
    # 加载数据集（固定种子保证公平对比）
    train_data, val_data = train_val_data_process(seed=args.seed, aug=args.aug, pt_norm=False, batch_size=args.batch_size)
    # 利用现有的模型进行模型的训练
    train_process = train_model_process(model, train_data, val_data, num_epochs=args.epochs,
                                        model_name=run_tag, lr=args.lr, use_cosine=args.cosine)
    matplot_acc_loss(train_process, f"train_curve_{run_tag}.png")