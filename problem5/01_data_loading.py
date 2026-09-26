"""Download FashionMNIST and build train, validation, and test DataLoaders."""
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data(batch_size=32, data_dir="data"):
    transform = transforms.ToTensor()
    train_all = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    train_set, validation_set = random_split(
        train_all, [55_000, 5_000], generator=torch.Generator().manual_seed(42)
    )
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(validation_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )


if __name__ == "__main__":
    train_loader, validation_loader, test_loader = load_data()
    print(f"Device: {get_device()}")
    print(f"Train: {len(train_loader.dataset)}; validation: {len(validation_loader.dataset)}; test: {len(test_loader.dataset)}")
