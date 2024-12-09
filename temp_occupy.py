import torch
import os

# 设置可见的GPU为2号

# 获取当前设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设你的 GPU 总显存是 80GB，分配 50GB 的显存 (50 / 80 = 0.625)
required_memory_fraction = 0.625
torch.cuda.set_per_process_memory_fraction(required_memory_fraction, 0)

# 分配一个大的张量来占用显存
# 张量大小 = (显存大小（字节） / 单精度浮点数(4字节))
# 50GB = 50 * 1024**3 字节 / 4 = 13421772800 元素
tensor_size = (13421772800,)  # 元素个数
tensor = torch.rand(tensor_size, dtype=torch.float32).to(device)

print(f"Allocated tensor of size {tensor_size} on {device}. Now occupying approximately 50GB of GPU memory.")

# 保持程序运行，防止张量被释放
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Process interrupted.")

