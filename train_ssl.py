import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.ham10000_dataset import HAM10000Dataset
from src.transforms.ssl_transforms import get_ssl_transforms
from src.models.encoder import ResNetEncoder
from src.models.projection_head import ProjectionHead
from src.losses.nt_xent import NTXentLoss


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # Dataset
    # -------------------------
    transform = get_ssl_transforms(image_size=224)

    dataset = HAM10000Dataset(
        csv_file="data/HAM10000/HAM10000_metadata.csv",
        image_dir="data/HAM10000",
        transform=transform,
        ssl=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=128,          # Good for T4 GPU
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # -------------------------
    # Models
    # -------------------------
    encoder = ResNetEncoder().to(device)
    projection_head = ProjectionHead(encoder.out_dim).to(device)

    # -------------------------
    # Loss
    # -------------------------
    criterion = NTXentLoss(temperature=0.5)

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(projection_head.parameters()),
        lr=1e-3
    )

    epochs = 100

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(epochs):
        total_loss = 0

        for (x1, x2) in dataloader:

            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass
            h1 = encoder(x1)
            h2 = encoder(x2)

            z1 = projection_head(h1)
            z2 = projection_head(h2)

            # Compute loss
            loss = criterion(z1, z2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    train()