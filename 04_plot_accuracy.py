"""Plot training and validation accuracy per epoch from history.json.

Run 03_train.py first to produce history.json.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

with open("history.json") as f:
    history = json.load(f)

n_epochs = len(history["train_metrics"])
epochs = np.arange(1, n_epochs + 1)

# Training accuracy is averaged over each epoch while the weights change, so
# like the book we shift it half an epoch left of the end-of-epoch
# validation accuracy
plt.plot(epochs - 0.5, history["train_metrics"], ".--", label="Training")
plt.plot(epochs, history["valid_metrics"], ".-", label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(epochs)
plt.grid()
plt.title("Fashion MNIST learning curves")
plt.legend()
plt.savefig("accuracy.png", dpi=150, bbox_inches="tight")
print("Saved accuracy.png")
