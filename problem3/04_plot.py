"""Plot training and validation accuracy over epochs from the history."""
import json

import matplotlib.pyplot as plt


def plot_accuracy(history, path="accuracy_curves.png"):
    epochs = range(1, len(history["train_metrics"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_metrics"], "o-", label="Training accuracy")
    ax.plot(epochs, history["valid_metrics"], "s-", label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("ImageClassifier on FashionMNIST")
    ax.set_xticks(list(epochs))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot to {path}")
    return fig


if __name__ == "__main__":
    with open("history.json") as f:
        history = json.load(f)
    plot_accuracy(history)
