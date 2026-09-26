"""Plot training and validation accuracy from history.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
HISTORY_PATH = SCRIPT_DIR / "history.json"
OUTPUT_PATH = SCRIPT_DIR / "accuracy.png"


def main():
    with HISTORY_PATH.open(encoding="utf-8") as history_file:
        history = json.load(history_file)

    epochs = range(1, len(history["train_accuracy"]) + 1)
    plt.plot(epochs, history["train_accuracy"], label="Training accuracy")
    plt.plot(epochs, history["valid_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved accuracy plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
