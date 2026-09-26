"""Train the ImageClassifier on Fashion MNIST with plain SGD.

Follows "Building an Image Classifier with PyTorch" in chapter 10 of
Hands-On Machine Learning with Scikit-Learn and PyTorch (Géron, 2025).
"""

import importlib
import json

import torch
import torch.nn as nn
import torchmetrics

# Module names start with a digit, so a plain import statement won't work
data = importlib.import_module("01_data_loading")
ImageClassifier = importlib.import_module("02_model").ImageClassifier

device = data.device
train_loader = data.train_loader
valid_loader = data.valid_loader


def evaluate_tm(model, data_loader, metric):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()


def train(model, optimizer, criterion, metric, train_loader, valid_loader,
          n_epochs):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_tm(model, valid_loader, metric).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train accuracy: {history['train_metrics'][-1]:.4f}, "
              f"valid accuracy: {history['valid_metrics'][-1]:.4f}")
    return history


if __name__ == "__main__":
    n_epochs = 20
    learning_rate = 0.1

    torch.manual_seed(42)
    model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300,
                            n_hidden2=100, n_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    xentropy = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task="multiclass",
                                     num_classes=10).to(device)

    print(f"Training on {device}")
    history = train(model, optimizer, xentropy, accuracy, train_loader,
                    valid_loader, n_epochs)

    torch.save(model.state_dict(), "image_classifier.pt")
    with open("history.json", "w") as f:
        json.dump(history, f, indent=2)
