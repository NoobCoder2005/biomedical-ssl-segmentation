import torch

from src.models.encoder import ResNetEncoder
from src.models.projection_head import ProjectionHead

encoder = ResNetEncoder()
proj = ProjectionHead(encoder.out_dim)

x = torch.randn(4, 3, 224, 224)

features = encoder(x)
z = proj(features)

print(features.shape)
print(z.shape)

