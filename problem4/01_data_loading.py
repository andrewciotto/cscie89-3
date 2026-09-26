"""Download FashionMNIST and prepare train, validation, and test loaders."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


BATCH_SIZE = 32
DATA_DIR = "data"


def get_device():
    """Return the fastest supported device available on this machine."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    transform = transforms.ToTensor()

    train_valid_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        transform=transform,
        download=True,
    )

    train_dataset, valid_dataset = random_split(
        train_valid_dataset,
        [55_000, 5_000],
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = get_device()
    print(f"Device: {device}")
    print(
        "Dataset sizes: "
        f"train={len(train_dataset)}, "
        f"valid={len(valid_dataset)}, "
        f"test={len(test_dataset)}"
    )
    print(
        "DataLoader batches: "
        f"train={len(train_loader)}, "
        f"valid={len(valid_loader)}, "
        f"test={len(test_loader)}"
    )


if __name__ == "__main__":
    main()
