"""Define the fully connected FashionMNIST image classifier."""
import torch.nn as nn


class ImageClassifier(nn.Module):
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


if __name__ == "__main__":
    print(ImageClassifier())
