import os
from medmnist import INFO, DermaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

data_root = "/blue/bme6938/haibinfan/project2/data"

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = DermaMNIST(
    split="train",
    transform=transform,
    download=True,
    size=224,
    root=data_root
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Data downloaded to:", data_root)

