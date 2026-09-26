"""Train the FashionMNIST ImageClassifier and record epoch metrics."""

import json

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


DATA_DIR = "data"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.01
SPLIT_SEED = 42


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_classifier_class():
    """Load ImageClassifier from 02_model.py, whose name is not an identifier."""
    model_path = Path(__file__).with_name("02_model.py")
    spec = spec_from_file_location("fashion_mnist_model", model_path)
    model_module = module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return model_module.ImageClassifier


def make_data_loaders():
    transform = transforms.ToTensor()
    train_valid_dataset = datasets.FashionMNIST(
        root=DATA_DIR, train=True, transform=transform, download=True
    )
    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR, train=False, transform=transform, download=True
    )
    train_dataset, valid_dataset = random_split(
        train_valid_dataset,
        [55_000, 5_000],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )
    return (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False),
    )


def accuracy_counts(logits, labels):
    predictions = logits.argmax(dim=1)
    return (predictions == labels).sum().item(), labels.size(0)


def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            batch_correct, batch_total = accuracy_counts(model(images), labels)
            correct += batch_correct
            total += batch_total
    return correct / total


def main():
    device = get_device()
    train_loader, valid_loader, _test_loader = make_data_loaders()

    ImageClassifier = load_classifier_class()
    model = ImageClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "train_accuracy": [], "valid_accuracy": []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            batch_correct, batch_total = accuracy_counts(logits, labels)
            correct += batch_correct
            total += batch_total

        mean_train_loss = total_loss / total
        train_accuracy = correct / total
        valid_accuracy = evaluate_accuracy(model, valid_loader, device)

        history["train_loss"].append(mean_train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["valid_accuracy"].append(valid_accuracy)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"train_loss={mean_train_loss:.4f} "
            f"train_accuracy={train_accuracy:.4f} "
            f"valid_accuracy={valid_accuracy:.4f}"
        )

    with open("history.json", "w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)

    return history


if __name__ == "__main__":
    main()
