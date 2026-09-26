"""Load Fashion MNIST and build train/valid/test DataLoaders.

Follows "Building an Image Classifier with PyTorch" in chapter 10 of
Hands-On Machine Learning with Scikit-Learn and PyTorch (Géron, 2025).
"""

import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Convert PIL images to float32 tensors scaled to [0, 1]
toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=toTensor)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=toTensor)

torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000])

torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"Train: {len(train_data):,}  Valid: {len(valid_data):,}  "
          f"Test: {len(test_data):,}")
    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch X: {tuple(X_batch.shape)} {X_batch.dtype}, "
          f"y: {tuple(y_batch.shape)} {y_batch.dtype}")
    print(f"Classes: {train_and_valid_data.classes}")
