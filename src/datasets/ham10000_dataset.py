# DATA PIPELINING LAYER
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, ssl=False):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.ssl = ssl

        # Assign index to each unique label
        self.classes = sorted(self.data['dx'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id = self.data.iloc[idx]['image_id']
        image_name = image_id + ".jpg"

        # Search in both Kaggle image folders
        part1_path = os.path.join(self.image_dir, "HAM10000_images_part_1", image_name)
        part2_path = os.path.join(self.image_dir, "HAM10000_images_part_2", image_name)

        if os.path.exists(part1_path):
            image_path = part1_path
        elif os.path.exists(part2_path):
            image_path = part2_path
        else:
            raise FileNotFoundError(f"{image_name} not found in dataset folders.")

        image = Image.open(image_path).convert("RGB")

        if self.ssl:
            view1 = self.transform(image)
            view2 = self.transform(image)
            return view1, view2
        else:
            label = self.class_to_idx[self.data.iloc[idx]['dx']]
            if self.transform:
                image = self.transform(image)
            return image, label


# TEST BLOCK
if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.ToTensor()

    dataset = HAM10000Dataset(
        csv_file="data/HAM10000/HAM10000_metadata.csv",
        image_dir="data/HAM10000",
        transform=transform
    )

    print("Dataset size:", len(dataset))

    image, label = dataset[0]
    print("Image shape:", image.shape)
    print("Label:", label)
