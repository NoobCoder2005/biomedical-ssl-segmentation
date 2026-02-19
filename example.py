import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

x = torch.rand(3, 256, 256)
print("Tensor shape:", x.shape)

