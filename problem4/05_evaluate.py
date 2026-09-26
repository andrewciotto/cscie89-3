"""Evaluate the trained FashionMNIST classifier on the held-out test set."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from importlib.util import module_from_spec, spec_from_file_location


DATA_DIR = "data"
BATCH_SIZE = 32
MODEL_PATH = Path(__file__).resolve().parent / "fashion_mnist_model.pth"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_classifier_class():
    model_path = Path(__file__).with_name("02_model.py")
    spec = spec_from_file_location("fashion_mnist_model", model_path)
    model_module = module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return model_module.ImageClassifier


def main():
    device = get_device()
    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ImageClassifier = load_classifier_class()
    model = ImageClassifier().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
