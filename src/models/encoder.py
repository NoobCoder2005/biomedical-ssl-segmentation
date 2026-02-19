import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self, base_model="resnet18"):
        super().__init__()

        # Updated API (no deprecation warning)
        resnet = getattr(models, base_model)(weights=None)

        # Remove final classification layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        self.out_dim = resnet.fc.in_features

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)  # cleaner than view
        return x
