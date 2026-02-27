import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        # Simple decoder (you can upgrade later to UNet style)
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder.out_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, x):
        features = self.encoder(x)

        # If encoder outputs flattened vector, reshape first
        if len(features.shape) == 2:
            features = features.unsqueeze(-1).unsqueeze(-1)

        out = self.decoder(features)

        # Upsample to input size
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out