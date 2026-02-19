from torchvision import transforms

def get_ssl_transforms(image_size=224):#for having equal sized images
    #Randomly changing these
    color_jitter = transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    )

    ssl_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),#Crop the image
        transforms.RandomHorizontalFlip(),#Flip left right randomly
        transforms.RandomApply([color_jitter], p=0.8),#randomly applying color jitters
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor()
    ])

    return ssl_transform
