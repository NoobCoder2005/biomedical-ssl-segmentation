import os
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
        csv_file="/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv",
        image_dir="/kaggle/input/skin-cancer-mnist-ham10000",
        transform=transform,
        ssl=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # -------------------------
    # Models
    # -------------------------
    encoder = ResNetEncoder().to(device)
    projection_head = ProjectionHead(encoder.out_dim).to(device)

    encoder.train()
    projection_head.train()

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
    start_epoch = 0

    # -------------------------
    # Resume from latest checkpoint (if exists)
    # -------------------------
    checkpoints = [f for f in os.listdir() if f.startswith("ssl_checkpoint_epoch_")]

    if checkpoints:
        latest_checkpoint = sorted(
            checkpoints,
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )[-1]

        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(start_epoch, epochs):
        total_loss = 0

        for (x1, x2) in dataloader:

            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad()

            h1 = encoder(x1)
            h2 = encoder(x2)

            z1 = projection_head(h1)
            z2 = projection_head(h2)

            loss = criterion(z1, z2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        # -------------------------
        # Save checkpoint every 10 epochs
        # -------------------------
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"ssl_checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'projection_head_state_dict': projection_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    # -------------------------
    # Save final model
    # -------------------------
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'projection_head_state_dict': projection_head.state_dict()
    }, "ssl_final_model.pth")

    print("Training completed and final model saved.")


if __name__ == "__main__":
    train()