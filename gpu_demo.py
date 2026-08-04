"""
Kaggle GPU 环境验证 demo
运行方式：在 Kaggle Notebook 中新建 cell，粘贴以下代码运行
预期：显示 GPU 型号（如 Tesla T4），loss 逐步下降
"""
import torch
import torch.nn as nn

# 1. 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")

# 2. 在 GPU 上训练一个简单的线性模型
model = nn.Linear(10, 1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

x = torch.randn(256, 10, device=device)
y = torch.randn(256, 1, device=device)

for epoch in range(50):
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch + 1:3d}  loss = {loss.item():.6f}")

print("Demo 运行完成 ✅")