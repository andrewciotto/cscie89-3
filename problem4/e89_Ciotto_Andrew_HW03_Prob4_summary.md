# HW03 Problem 4 Session Summary

## What we built

This session added a step-by-step PyTorch FashionMNIST classifier workflow:

- `01_data_loading.py` downloads the torchvision FashionMNIST training and test sets, converts images to tensors, splits the 60,000 training examples into 55,000 training and 5,000 validation examples, and builds batch-size-32 DataLoaders. Device selection prefers CUDA, then MPS, then CPU.
- `02_model.py` defines `ImageClassifier`, which flattens each 28×28 grayscale image and applies fully connected layers of 300 and 100 units with ReLU activations, followed by 10 output logits.
- `03_train.py` trains with `CrossEntropyLoss` and plain SGD for 20 epochs. It records mean training loss, training accuracy, and validation accuracy in a history dictionary and prints the metrics each epoch. It writes the history to `history.json` and model weights to `fashion_mnist_model.pth`.
- `04_plot_accuracy.py` reads the saved history and plots training and validation accuracy against epoch, with axis labels and a legend, saving the plot to `accuracy.png`.
- `05_evaluate.py` loads the saved weights and reports accuracy on the separate 10,000-example FashionMNIST test set.

## Key decisions

- The test dataset remains separate from the train/validation split and is used only by the evaluation script.
- The training split in `03_train.py` uses a fixed seed (42) for a repeatable split.
- Accuracy uses the predicted class from `argmax` over the model's logits, avoiding an extra `torchmetrics` dependency.
- The model and training scripts have numeric filename prefixes, so the training and evaluation scripts load `ImageClassifier` from `02_model.py` through Python's importlib utilities.
- The scripts save history and weights beside the scripts so the plotting and evaluation scripts can read their outputs.

## Issues and workflow notes

- The repository is rooted one directory above the working project folder. Updating the Git index and creating commits required elevated access because the sandbox permits writes inside `problem4` but not to the parent repository's `.git` directory.
- The existing modification to `problem3/e89_Ciotto_Andrew_HW03_Prob3.ipynb` was present during this work and was left untouched.
- Git reports `main...origin/main [gone]` in the local status, although the `origin` remote is configured as `https://github.com/andrewciotto/cscie89-3.git`. This summary commit still needs to be pushed to `origin/main`.
- The classifier training and plotting/evaluation scripts were created but not run during this session; running training downloads the dataset and generates the history and model checkpoint required by the later scripts.

## Commits created during the session

- `1ed2ca3` — Add FashionMNIST data loading script
- `5dee6cf` — Add FashionMNIST image classifier model
- `6935344` — Add FashionMNIST training loop
- `4fe467a` — Add FashionMNIST accuracy plot
- `2889c0a` — Add FashionMNIST test evaluation
