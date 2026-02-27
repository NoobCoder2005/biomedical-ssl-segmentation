import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class HAM10000Segmentation(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=224):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Get all image filenames
        self.images = [
            f for f in os.listdir(image_dir)
            if f.endswith(".jpg")
        ]

        self.transform_image = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

        self.transform_mask = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        image_path = os.path.join(self.image_dir, img_name)

        # Build corresponding mask name
        mask_name = img_name.replace(".jpg", "_segmentation.png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load mask
        mask = Image.open(mask_path).convert("L")

        image = self.transform_image(image)
        mask = self.transform_mask(mask)

        # Convert mask to binary (0 or 1)
        mask = (mask > 0).float()

        return image, mask