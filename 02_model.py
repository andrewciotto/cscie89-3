"""Define the ImageClassifier MLP for Fashion MNIST.

Follows "Building an Image Classifier with PyTorch" in chapter 10 of
Hands-On Machine Learning with Scikit-Learn and PyTorch (Géron, 2025).
"""

import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
    def __init__(self, n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100,
                 n_classes=10):
        super().__init__()
        # Outputs raw logits: nn.CrossEntropyLoss applies softmax internally
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes)
        )

    def forward(self, X):
        return self.mlp(X)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = ImageClassifier()
    print(model)
    X_dummy = torch.rand(32, 1, 28, 28)
    print(f"Output shape: {tuple(model(X_dummy).shape)}")
    n_params = sum(param.numel() for param in model.parameters())
    print(f"Parameters: {n_params:,}")
