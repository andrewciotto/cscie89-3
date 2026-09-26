"""Evaluate the trained ImageClassifier on the Fashion MNIST test set.

Run 03_train.py first to produce image_classifier.pt.
"""

import importlib

import torch
import torchmetrics

# Module names start with a digit, so a plain import statement won't work
data = importlib.import_module("01_data_loading")
ImageClassifier = importlib.import_module("02_model").ImageClassifier
evaluate_tm = importlib.import_module("03_train").evaluate_tm

device = data.device

model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100,
                        n_classes=10).to(device)
model.load_state_dict(torch.load("image_classifier.pt", map_location=device))

accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
test_accuracy = evaluate_tm(model, data.test_loader, accuracy).item()

print(f"Test set: {len(data.test_data):,} images")
print(f"Test accuracy: {test_accuracy:.4f}")
