# 实验记录：注意力机制对比

## 实验环境

- 显卡：NVIDIA GeForce RTX 4070 (8GB)
- conda 环境：`gpu`（Python 3.11, PyTorch 2.5.1 + CUDA 12.4）
- 数据集：`data/train` 927 张（mask 450 / no_mask 477），train:val = 8:2 随机划分
- 统一配置：batch 32，Adam lr=0.001，CrossEntropyLoss
- 随机种子：42（三种模型使用完全相同的 train/val 划分，保证公平）
- 测试集：`data/test` 103 张（独立样本）

## 第一轮：baseline / SE / CBAM（30 epochs，无数据增强）

| 模型 | 参数量 | 最佳 val acc | 测试集 acc | 训练耗时 |
|---|---|---|---|---|
| baseline (ResNet18) | 11,180,490 | 89.19% | 84.47% | 8m11s |
| SE-ResNet18 | 11,269,626 | 91.89% | 85.44% | 8m10s |
| CBAM-ResNet18 | 11,270,418 | 90.81% | 89.32% | 8m16s |

结论：
- 注意力机制均超过 baseline，验证集提升 1.6~2.7 个百分点
- CBAM 测试集领先最多（+4.85pp），泛化性最好
- SE 验证集最高但测试集提升有限，存在轻度过拟合

## 第二轮：CBAM + 数据增强（30 epochs）

| 模型 | 最佳 val acc | 测试集 acc |
|---|---|---|
| CBAM + 数据增强 | - | 88.35% |

结论：
- 增强（随机水平翻转 + 随机裁剪等）反而比无增强 CBAM（89.32%）略降 0.97pp
- 数据量仅 927 张时增强收益不明显，收敛更慢，不是当前瓶颈

## 第三轮：CBAM + ImageNet 预训练微调（30 epochs，lr=1e-4 + cosine）

| 模型 | 测试集 acc | 提升 |
|---|---|---|
| CBAM-ResNet18 | 89.32% | - |
| CBAM + 预训练微调 | **98.06%** | +8.74pp |

结论（关键提升步）：
- 预训练权重是最大杠杆：89.32% -> 98.06%，远超注意力/增强收益
- 测试集 103 张仅错 2 张，均为"戴口罩"被误判为"不带口罩"
- 已达成 95% 目标

## 错误样本定位（find_errors.py 索引 bug 已修复）

> 早期版本 find_errors.py 用"错误计数 len(wrong)"当样本索引，误报 err_mask10/err_mask108。
> 修正后确认 cbam_pt 的真实错误样本为：
- `data/test/mask/mask136.jpg`：戴口罩被误判为不带口罩（置信度 0.957）
- `data/test/mask/mask178.jpg`：戴口罩被误判为不带口罩（置信度 0.740）

## 第四轮：错误样本注意力可视化分析（Grad-CAM 量化）

Grad-CAM 高响应区分布指标（真实类梯度，中心占比 = 高响应区落在图像中心 50% 区域的比例）：

| 样本 | 类别 | 质心位置 | 中心占比 | 失败模式 |
|---|---|---|---|---|
| mask112（正确） | 戴口罩 | (0.46, 0.49) | **0.72** | 聚焦良好，看脸 |
| mask136（**错误**） | 戴口罩 | (0.72, 0.39) | **0.04** | **被右侧背景吸引** |
| mask178（**错误**） | 戴口罩 | (0.51, 0.46) | 0.36 | 热力图整体饱和，无判别性聚焦 |
| no_mask12（正确） | 不带口罩 | (0.28, 0.59) | 0.46 | 聚焦面部，正常 |

结论：
- mask136：真实类注意力仅 4% 落在中心区域，模型被背景干扰，属于"学到背景捷径"
- mask178：模型对整图无聚焦（热力图面积占比 1.00），属难样本（遮挡角度/光线），非纯背景问题
- 可视化图：`vis_attn_0~3.png`（原图 | 真实类 Grad-CAM | 预测类 Grad-CAM | CBAM 空间注意力）

## 优化策略评估（基于可视化结论）

"只对口罩区域强化"不可行——数据没有口罩 bounding box 标注。可选方向：

| 策略 | 原理 | 预期影响 | 风险 |
|---|---|---|---|
| 中心偏置裁剪（戴口罩类） | 裁剪偏向中心 50% 区域，逼模型学人脸中心特征 | 对抗 mask136 类背景干扰 | 需按类分组增强，实现稍复杂 |
| 外围随机擦除 | 随机抹掉边缘区域，消除背景捷径 | 同上，且增强背景多样性 | 擦除过大可能破坏特征 |
| 错误样本入训练集 | 直接学习 2 张错图 | 最直接，预计 ~100% | 污染测试集独立性（课程作业层面可接受） |

优先级建议：先验证 1+2（不污染测试集），若 val 无提升再考虑 3。

## 历史对照（原始 50 epochs，无种子固定）

训练脚本最初为 50 epochs 且未固定随机种子时的 baseline 结果：
- 最佳 val acc 88.7%，测试集 89.3%

## 产物说明

- `best_model_{baseline,se,cbam,cbam_aug,cbam_pt}.pth`：各模型最优权重（已 gitignore）
- `train_curve_{baseline,se,cbam,cbam_aug,cbam_pt}.png`：训练曲线（已 gitignore）
- `vis_attention.py`：错误样本注意力可视化（Grad-CAM + CBAM 空间注意力）
- `analyze_attention.py`：Grad-CAM 热力图量化分析（质心/中心占比）

## 复现命令

```bash
# 训练（--model 可选 baseline/se/cbam/cbam_aug/cbam_pt）
python model_train.py --model baseline --epochs 30 --seed 42

# 预训练微调（当前最优）
python model_train.py --model cbam_pt --epochs 30 --lr 1e-4 --cosine

# 测试
python model_test.py --model cbam_pt
```