"""Train the ImageClassifier on FashionMNIST with plain SGD."""
import importlib
import json

import torch
import torch.nn as nn
import torchmetrics

data = importlib.import_module("01_data")
ImageClassifier = importlib.import_module("02_model").ImageClassifier

N_EPOCHS = 20
LEARNING_RATE = 0.002


def evaluate_tm(model, data_loader, metric, device):
    """Compute a torchmetrics metric over a whole DataLoader."""
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute().item()


def train(model, optimizer, loss_fn, metric, train_loader, valid_loader,
          n_epochs, device):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.0
        metric.reset()
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_tm(model, valid_loader, metric, device))
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train accuracy: {history['train_metrics'][-1]:.4f}, "
              f"valid accuracy: {history['valid_metrics'][-1]:.4f}")
    return history


if __name__ == "__main__":
    device = data.get_device()
    print(f"Using device: {device}")
    train_loader, valid_loader, _ = data.get_loaders()

    torch.manual_seed(42)
    model = ImageClassifier().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    xentropy = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

    history = train(model, optimizer, xentropy, accuracy, train_loader,
                    valid_loader, N_EPOCHS, device)

    torch.save(model.state_dict(), "image_classifier.pt")
    with open("history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Saved model to image_classifier.pt and history to history.json")
