from src.datasets.ham10000_dataset import HAM10000Dataset
from src.transforms.ssl_transforms import get_ssl_transforms

transform = get_ssl_transforms()

dataset = HAM10000Dataset(
    csv_file="data/HAM10000/metadata.csv",
    image_dir="data/HAM10000/images",
    transform=transform,
    ssl=True
)

view1, view2 = dataset[0]
print(view1.shape, view2.shape)

