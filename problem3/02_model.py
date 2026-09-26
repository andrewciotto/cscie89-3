"""Define the ImageClassifier MLP for FashionMNIST."""
import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
    def __init__(self, n_inputs=28 * 28, n_hidden1=300, n_hidden2=100,
                 n_classes=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes),  # logits, no final activation
        )

    def forward(self, X):
        return self.mlp(X)


if __name__ == "__main__":
    model = ImageClassifier()
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_params:,}")
    logits = model(torch.rand(32, 1, 28, 28))
    print(f"Output shape: {tuple(logits.shape)}")
