# HW03 Problem 5 Session Summary

## Goal

Build a PyTorch FashionMNIST image classifier following the Chapter 10 “Building an Image Classifier with PyTorch” workflow from Géron (2025), with five numbered scripts and a combined Jupyter notebook.

## Work completed

- `01_data_loading.py` downloads FashionMNIST through `torchvision`, converts images with `transforms.ToTensor()`, uses a deterministic 55,000/5,000 split of the official 60,000-image training set, preserves the official 10,000-image test set, builds batch-size-32 DataLoaders, and selects CUDA, MPS, or CPU in that order.
- `02_model.py` defines `ImageClassifier`, which flattens 28×28 inputs, applies hidden layers of 300 and 100 units with ReLU, and returns ten logits.
- `03_train.py` trains with `nn.CrossEntropyLoss` and plain SGD for 20 epochs. It records training loss, training accuracy, and validation accuracy per epoch, then saves weights to `model.pth` and metrics to `history.json`.
- `04_plot_accuracy.py` plots training and validation accuracy with labeled axes and a legend, saving `accuracy.png`.
- `05_evaluate.py` loads the saved weights, evaluates on the separate test set, and reports final test accuracy.
- `e89_Ciotto_Andrew_HW03_Prob5.ipynb` contains the same workflow in order, with explanatory markdown and notebook-ready code cells.

## Verification and limits

Python syntax parsing succeeded for all five scripts and every notebook code cell. The 20-epoch training run was not executed, so no model weights, history, plot, or measured test accuracy were generated in this session.

## Git

The five scripts and notebook were committed individually on `main` in these commits:

- `2ad9b36` Add FashionMNIST Problem 5 data loading
- `deedf6c` Add FashionMNIST Problem 5 classifier model
- `d11b997` Add FashionMNIST Problem 5 training loop
- `746ca78` Add FashionMNIST Problem 5 accuracy plot
- `3d33684` Add FashionMNIST Problem 5 test evaluation
- `76c831b` Add combined notebook for HW03 Problem 5

At the start of this summary task, `main` was six commits ahead of `origin/main`. A pre-existing modification to `problem3/e89_Ciotto_Andrew_HW03_Prob3.ipynb` was left untouched.
