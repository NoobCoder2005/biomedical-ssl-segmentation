import torch
from src.losses.nt_xent import NTXentLoss

loss_fn = NTXentLoss()

z1 = torch.randn(4, 128)
z2 = torch.randn(4, 128)

loss = loss_fn(z1, z2)
print("Loss:", loss.item())

