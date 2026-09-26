"""Fully connected classifier for 28x28 grayscale images."""

from torch import nn


class ImageClassifier(nn.Module):
    """Classify flattened 28x28 images into 10 classes."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, images):
        return self.network(images)
