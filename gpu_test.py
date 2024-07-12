import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA is available! PyTorch can use GPU.")
else:
    print("CUDA is not available. Using CPU.")

# 创建一个张量并移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(3, 3).to(device)

# 检查张量是否在GPU上
if x.is_cuda:
    print("Tensor is on GPU.")
else:
    print("Tensor is on CPU.")
