"""Plot training and validation accuracy stored by 03_train.py."""
import json
import matplotlib.pyplot as plt


def plot_accuracy(history_file="history.json", output_file="accuracy.png"):
    with open(history_file, encoding="utf-8") as f:
        history = json.load(f)
    epochs = range(1, len(history["training_accuracy"]) + 1)
    plt.plot(epochs, history["training_accuracy"], label="Training accuracy")
    plt.plot(epochs, history["validation_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


if __name__ == "__main__":
    plot_accuracy()
