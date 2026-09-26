"""Evaluate the saved classifier on the separate FashionMNIST test set."""
import importlib.util
from pathlib import Path
import torch


def _load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data_module = _load_module("01_data_loading.py", "data_loading")
get_device, load_data = data_module.get_device, data_module.load_data
ImageClassifier = _load_module("02_model.py", "model").ImageClassifier


def evaluate(checkpoint="model.pth", batch_size=32):
    device = get_device()
    _, _, test_loader = load_data(batch_size=batch_size)
    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    result = correct / total
    print(f"Test accuracy: {result:.4%}")
    return result


if __name__ == "__main__":
    evaluate()
