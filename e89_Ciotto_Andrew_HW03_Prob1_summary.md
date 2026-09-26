# HW03 Problem 1 — Fashion MNIST Image Classifier: Session Summary

This session built a Fashion MNIST image classifier in PyTorch, following the
"Building an Image Classifier with PyTorch" section of chapter 10 of
*Hands-On Machine Learning with Scikit-Learn and PyTorch* (Géron, 2025) and its
companion notebook `10_neural_nets_with_pytorch.ipynb`. Each step is a
separate script, committed to git and pushed to
[andrewciotto/cscie89-3](https://github.com/andrewciotto/cscie89-3).

## What we built

| Script | Purpose | Commit |
|---|---|---|
| `01_data_loading.py` | Downloads Fashion MNIST with torchvision, turns images into tensors, splits the 60,000 training images into 55,000 train and 5,000 validation, keeps the 10,000-image test set separate, builds batch-32 DataLoaders, and picks the device (cuda → mps → cpu) | `3203f28` |
| `02_model.py` | `ImageClassifier` `nn.Module`: Flatten → Linear(784, 300) → ReLU → Linear(300, 100) → ReLU → Linear(100, 10), returning raw logits | `cded074` |
| `03_train.py` | Training loop with `nn.CrossEntropyLoss`, torchmetrics multiclass accuracy and plain SGD for 20 epochs. It records mean train loss, train accuracy and validation accuracy in a history dictionary | `4c94fa9` |
| `04_plot_accuracy.py` | Plots training and validation accuracy per epoch with matplotlib and saves `accuracy.png` | `0dc735a` |
| `accuracy.png` | The learning-curve plot from the 20-epoch run | `f04f565` |
| `05_evaluate.py` | Loads the trained weights and reports accuracy on the held-out test set | `7ee4e17` |

A `.gitignore` excludes `datasets/`, `__pycache__/` and `*.pt`.

To reproduce, run the scripts in order: `03_train.py` produces
`image_classifier.pt` and `history.json`, which `04_plot_accuracy.py` and
`05_evaluate.py` need.

## Results

- **Device:** Apple Silicon GPU (`mps`). Training for 20 epochs took about 70 seconds.
- **Model size:** 266,610 trainable parameters, the same number the book reports.
- **Final epoch (20):** train loss 0.1869, train accuracy 0.9283, validation accuracy 0.8826.
- **Best validation accuracy:** 0.8900, at epoch 18.
- **Test accuracy:** **0.8785** on the 10,000 held-out test images.

Test accuracy is close to validation accuracy, so the 5,000-image validation
set gave a reliable estimate of how the model does on new data.

## Key decisions

- **Stayed close to the book.** Same transform
  (`T.ToImage()` + `T.ToDtype(torch.float32, scale=True)`), same
  `torch.manual_seed(42)` before the split and before building the model, same
  architecture, and the same learning rate (SGD, lr = 0.1, no momentum). This
  makes the results easy to compare with the text.
- **The model returns logits with no softmax.** `nn.CrossEntropyLoss` applies
  log-softmax itself, so a softmax layer in the model would be redundant and
  less numerically stable.
- **Only the training loader shuffles.** The validation and test loaders keep
  a fixed order.
- **One file per step.** Later scripts reuse earlier ones instead of copying
  code: `05_evaluate.py` imports the data loaders, the model class and the
  `evaluate_tm` function from `03_train.py`. The `if __name__ == "__main__":`
  guards stop an import from running training or printing demo output.
- **Training saves its outputs.** `03_train.py` writes the weights
  (`image_classifier.pt`) and the history (`history.json`), so plotting and
  evaluation don't need to retrain.
- **Training accuracy is shifted half an epoch in the plot.** Training
  accuracy is averaged over each epoch while the weights are still changing,
  but validation accuracy is measured at the end of the epoch. Plotting
  training points at epoch − 0.5 (as the book does) lines the two curves up
  fairly.
- **Generated files mostly stay out of git.** The dataset and model weights
  are ignored, and `history.json` is left untracked. `accuracy.png` was
  committed because the assignment asked for the plot.

## Issues and observations

- **Script names start with a digit.** Files like `01_data_loading.py` can't
  be loaded with a normal `import` statement, so the later scripts use
  `importlib.import_module("01_data_loading")`.
- **torchmetrics was missing.** It wasn't installed in the miniconda
  Python 3.14 environment, so it was added with pip (version 1.9.0).
  torch 2.14.0, torchvision 0.29.0 and matplotlib 3.11.2 were already
  installed.
- **Pushes were sometimes out of sync.** The first three commits were already
  on GitHub before the first push in this session, so that push only sent the
  newer commits. Pushes were done only when asked.
- **Validation accuracy jumps around.** It dropped noticeably at epochs 6 and
  14 (0.8574 and 0.8644). This is typical for plain SGD with a fairly high
  learning rate and no momentum.
- **Early overfitting.** By epoch 20, training accuracy (0.928) is about 4.5
  points above validation accuracy (0.883), and validation accuracy stopped
  improving after about epoch 11. Early stopping, a lower or decaying learning
  rate, momentum, or regularization (dropout, weight decay) could help in
  later work.
