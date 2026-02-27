from src.datasets.ham10000_segmentation import HAM10000Segmentation

dataset = HAM10000Segmentation(
    image_dir="data/HAM10000/images",
    mask_dir="data/HAM10000/masks"
)

print("Dataset size:", len(dataset))

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

