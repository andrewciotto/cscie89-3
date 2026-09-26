"""Train ImageClassifier with SGD and record epoch metrics."""
import json
import importlib.util
from pathlib import Path
import torch
from torch import nn


def _load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


get_device, load_data = (lambda m: (m.get_device, m.load_data))(_load_module("01_data_loading.py", "data_loading"))
ImageClassifier = _load_module("02_model.py", "model").ImageClassifier


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_model(epochs=20, batch_size=32, learning_rate=0.01, checkpoint="model.pth", history_file="history.json"):
    device = get_device()
    train_loader, validation_loader, _ = load_data(batch_size=batch_size)
    model = ImageClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    history = {"training_loss": [], "training_accuracy": [], "validation_accuracy": []}

    for epoch in range(epochs):
        model.train()
        loss_sum = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            n = labels.size(0)
            loss_sum += loss.item() * n
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += n
        history["training_loss"].append(loss_sum / total)
        history["training_accuracy"].append(correct / total)
        history["validation_accuracy"].append(accuracy(model, validation_loader, device))
        print(f"Epoch {epoch + 1:02d}/{epochs}: loss={history['training_loss'][-1]:.4f}, "
              f"train_acc={history['training_accuracy'][-1]:.4f}, "
              f"val_acc={history['validation_accuracy'][-1]:.4f}")

    torch.save(model.state_dict(), checkpoint)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return history


if __name__ == "__main__":
    train_model()
