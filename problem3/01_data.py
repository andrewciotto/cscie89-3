"""Load FashionMNIST, split train/valid/test, and build DataLoaders."""
import torch
import torchvision
import torchvision.transforms.v2 as T

BATCH_SIZE = 32


def get_device():
    """Return the best available device: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_datasets(root="datasets"):
    """Download FashionMNIST and split the 60,000 training images 55,000/5,000."""
    toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    train_and_valid_data = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=True, transform=toTensor)
    test_data = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=True, transform=toTensor)
    torch.manual_seed(42)
    train_data, valid_data = torch.utils.data.random_split(
        train_and_valid_data, [55_000, 5_000])
    return train_data, valid_data, test_data


def get_loaders(batch_size=BATCH_SIZE):
    train_data, valid_data, test_data = get_datasets()
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
    train_loader, valid_loader, test_loader = get_loaders()
    print(f"Train: {len(train_loader.dataset)}, "
          f"Valid: {len(valid_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")
    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch shapes: X={tuple(X_batch.shape)}, y={tuple(y_batch.shape)}")
