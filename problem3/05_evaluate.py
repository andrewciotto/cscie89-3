"""Evaluate the trained ImageClassifier on the held-out test set."""
import importlib

import torch
import torchmetrics

data = importlib.import_module("01_data")
ImageClassifier = importlib.import_module("02_model").ImageClassifier
evaluate_tm = importlib.import_module("03_train").evaluate_tm


if __name__ == "__main__":
    device = data.get_device()
    _, _, test_loader = data.get_loaders()

    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load("image_classifier.pt", map_location=device))
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

    test_acc = evaluate_tm(model, test_loader, accuracy, device)
    print(f"Test accuracy: {test_acc:.4f} ({test_acc:.2%})")
