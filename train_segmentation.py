import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.encoder import ResNetEncoder
from src.models.segmentation_model import SegmentationModel
from src.datasets.ham10000_segmentation import HAM10000Segmentation


def load_ssl_encoder(checkpoint_path, device):
    encoder = ResNetEncoder().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    print("✅ SSL encoder loaded successfully")
    return encoder


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # Dataset
    # -------------------------
    dataset = HAM10000Segmentation(
        image_dir="data/HAM10000/images",
        mask_dir="data/HAM10000/masks"
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # -------------------------
    # Load SSL Encoder
    # -------------------------
    encoder = load_ssl_encoder(
        "models/ssl_final_model.pth",
        device
    )

    model = SegmentationModel(encoder).to(device)

    # Freeze encoder first (recommended)
    for param in model.encoder.parameters():
        param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5  # Start small

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(epochs):
        total_loss = 0

        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(dataloader):.4f}")

    print("🔥 Segmentation training complete")


if __name__ == "__main__":
    train()