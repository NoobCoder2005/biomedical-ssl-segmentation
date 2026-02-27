import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    def __init__(self, base_model="resnet18"):
        super().__init__()

        resnet = getattr(models, base_model)(weights=None)

        # Remove avgpool and fc
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        self.out_dim = 512  # for resnet18

    def forward(self, x):
        x = self.encoder(x)
        return x
