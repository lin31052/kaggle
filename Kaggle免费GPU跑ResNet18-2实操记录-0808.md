# Kaggle 免费 GPU 跑 ResNet18-2 实操记录（2026-08-08）

> 目标：本地没有 GPU，借 Kaggle 免费 T4 训练 `ResNet18-2`（大米 5 类分类，50 epochs）。
> 结果：已跑通，Epoch 2 即达到 train acc 98.4% / val acc 92.9%，加速后单 epoch 约 40 秒。
> 本文按实际操作顺序记录，可直接照做。

---

## 一、整体思路

```mermaid
flowchart LR
    A["本地改代码<br/>(源码/ResNet18-2)"] --> B["git push<br/>github.com/lin31052/kaggle"]
    B --> C["Kaggle Notebook<br/>git pull + unzip data.zip"]
    C --> D["T4 GPU 训练<br/>(nohup 后台 + tail 看日志)"]
```

**关键决策**：代码**和**数据都走 GitHub 中转（`data.zip` 74.8MB，未超 GitHub 单文件 100MB 限制）。
不用页面上传（浏览器文件对话框无法脚本化）、不用 SSH（**Kaggle Notebook 没有 SSH 服务**，`ssh`/`scp` 连不进去）。

---

## 二、第一步：代码最小改造（3 处路径）

原版代码在 Windows 上写死了路径，直接上 Kaggle（Linux）必崩。**只改路径，逻辑不动**：

| 文件 | 原代码（Windows 专用） | 改后（跨平台） |
|---|---|---|
| `model_train.py` | `ROOT_TRAIN = r'data\train'` | `ROOT_TRAIN = 'data/train'` |
| `model_train.py` | `torch.save(best_model_wts, "C:/Users/86159/Desktop/ResNet18-2/best_model.pth")` | `torch.save(best_model_wts, "best_model.pth")`（存当前目录） |
| `model_test.py` | `ROOT_TRAIN = r'data\test'` | `ROOT_TRAIN = 'data/test'` |

`model.py` 无需改动。

---

## 三、第二步：数据打包 + 推送

本地 `源码/ResNet18-2/data`（train 5485 张 + test 2396 张，共 25,221 个文件，87.9MB）打包：

```powershell
# 顶层必须是 data/，这样解压后 data/train 结构直接成立
Compress-Archive -Path '源码\ResNet18-2\data' -DestinationPath 'data.zip' -CompressionLevel Optimal
```

得到 `data.zip`（74.8MB），连同 5 个 .py 一起 commit + push 到中转仓库：

```bash
git add ResNet18-2/
git commit -m "添加 ResNet18-2 大米分类项目 + 训练数据包"
git push origin main
```

> 注意：GitHub 会提示文件超 50MB 推荐值，只是警告，<100MB 可正常存储。

---

## 四、第三步：Kaggle Notebook 操作（4 个 cell）

Kaggle 是临时容器（约 12 小时、长时间不活动会回收），**每次会话都要重跑 Cell 1**。

### Cell 1 · 拉代码 + 解压数据（`%%bash`）

```bash
%%bash
set -e
cd /kaggle/working
[ -d kaggle/.git ] || git clone https://github.com/lin31052/kaggle.git
cd /kaggle/working/kaggle/ResNet18-2
unzip -o -q data.zip
ls data/train
```

预期输出 5 个类目：`Arborio Basmati Ipsala Jasmine Karacadag`。

### Cell 2 · 验证 GPU 和数据（Python）

```python
import torch, os
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("当前目录:", os.getcwd())
print("训练集类目:", os.listdir('kaggle/ResNet18-2/data/train'))
```

预期：`CUDA: True Tesla T4` + 5 个类目。

### Cell 3 · 后台启动训练（`%%bash`）

```bash
%%bash
pkill -9 -f model_train.py 2>/dev/null
sleep 1
cd /kaggle/working/kaggle
git pull
cd ResNet18-2
nohup python -u model_train.py > train.log 2>&1 &
echo "PID=$!"
```

### Cell 4 · 轮询日志（`%%bash`，每 30~60 秒点一次）

```bash
%%bash
tail -8 /kaggle/working/kaggle/ResNet18-2/train.log
```

### 训练完成 · 汇总结果

```bash
%%bash
grep -E "Epoch|train loss|val loss|最佳验证" /kaggle/working/kaggle/ResNet18-2/train.log
```

---

## 五、加快训练（改动集中在 model_train.py，约 15 行）

| 手段 | 作用 | 收益 |
|---|---|---|
| `torch.cuda.amp.autocast` + `GradScaler`（AMP 混合精度） | FP16 前向/损失、FP32 权重，吃满 T4 Tensor Core | **~2x**（主要来源） |
| `torch.backends.cudnn.benchmark = True` | 输入固定 224×224，自动选最优卷积算法 | ~1.2x |
| `num_workers` 2 → 4 | 数据加载并行（Kaggle 4 vCPU） | 减少加载瓶颈 |

**踩坑记录：`torch.compile` 被移除**。刚开始加了它，结果 T4 上**首次编译极慢**（日志停在启动警告后 10+ 分钟无输出，`Not enough SMs to use max_autotune_gemm`），收益抵不过卡顿，最终去掉，只保留 AMP + benchmark。

提速后实测单 epoch 约 **40 秒**（第一轮含预热约 1 分钟），50 epochs 约 30~40 分钟。

---

## 六、日志显示：为什么"看不到进度" & 怎么解决

### 原因

原版代码 `print` 不带 flush，在 Kaggle notebook **前台跑 cell** 时，Jupyter 把 stdout 放进 IOPub 队列，**cell 结束才一次性刷出**——所以前台跑 50 个 epoch 期间页面毫无动静，像卡死。

### 解决：nohup 后台 + 日志落盘 + tail 轮询

1. `nohup python -u model_train.py > train.log 2>&1 &`——`-u` 强制无缓冲，每行立即写盘（不加 `-u` 日志会块缓冲，tail 半天不动）；
2. 训练 cell 立即返回，不阻塞其他 cell；
3. 用 `tail` cell **每 30~60 秒点一次运行**（不是实时滚动，能用就行）。

### 判断"进程是否活着"

```bash
%%bash
ps aux | grep model_train | grep -v grep
```

正常会出现 **3 个进程**：1 个主训练 + 2 个 DataLoader worker（`num_workers=2`），这是正常的，不是多个训练实例。

---

## 七、实测数据（2026-08-08）

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 0 | 0.1936 | 92.99% | 0.4600 | 79.74% |
| 2 | 0.0466 | 98.37% | 0.2052 | 92.91% |

首个 epoch 就 80% 验证精度，任务（5 类大米）特征差异明显，收敛很快。

---

## 八、踩坑清单（按遇到顺序）

1. **路径写死**：`r'data\train'` 和 `C:/Users/86159/...` 在 Linux 上必崩 → 改正斜杠 + 相对路径。
2. **Cell 工作目录不同**：`%%bash` 里的 `cd` 只对那个 cell 生效；Python cell 的 cwd 是 `/kaggle/working`，访问数据要写完整相对路径 `kaggle/ResNet18-2/data/...`。
3. **Kaggle working 目录会清空**：页面刷新/会话不活动后 `/kaggle/working` 重置，所以 Cell 1（clone+unzip）每次会话重跑。
4. **没有 SSH**：Kaggle Notebook 不提供 SSH，只有 Web 终端 / `%%bash` / 页面上传。
5. **页面上传不可脚本化**：Upload 按钮打开的是 OS 文件对话框，浏览器自动化无法操作 → 数据走 GitHub 中转。
6. **`torch.compile` 在 T4 卡编译**：首次编译极慢，移除，AMP 已足够。
7. **`python` 不加 `-u` 日志是块缓冲**：tail 会几行几行跳 → 必须 `python -u`。
8. **ps 看到 3 个 python 进程**：主进程 + 2 个 DataLoader worker，正常现象。
9. **弃用警告**（`torch.cuda.amp.*` deprecated）：功能不受影响，已在新版本换成 `torch.amp.GradScaler('cuda', ...)` / `torch.amp.autocast('cuda', ...)`。

---

## 九、产物

- 训练结束在 `/kaggle/working/kaggle/ResNet18-2/` 生成 `best_model.pth`（42MB），页面文件列表右键可下载；
- 本地测试：`python model_test.py`（加载 `best_model.pth` + `data/test`，输出测试准确率和单张预测）。

## 十、涉及提交（github.com/lin31052/kaggle）

| Commit | 说明 |
|---|---|
| `466a888` | 添加 ResNet18-2 代码（路径跨平台） |
| `f5fd195` | 添加 data.zip 数据包 |
| `e419305` | 训练提速：AMP + cudnn benchmark + torch.compile + num_workers=4 |
| `96b3bd1` | 消除 AMP API 弃用警告 |
| `836d671` | 移除 torch.compile（T4 首次编译过慢） |