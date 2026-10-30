# Replication Files for the paper "On The Localization Of Shortcuts In Vision Models"

This repository presents replication files for the paper "On The Localization Of Shortcuts In Vision Models" (2026)

## Replication Instuctions

1. Download MNIST, CIFAR-10, CUB-200-2011, Places365, and CelebA datasets
2. Generate the CIFAR-10 and Waterbirds datasets using "c-datasets-and-tables.ipynb"
3. Run "c-train.py" with appropriate flags to train the models
4. Run "c-val.py" with appropriate flags to evaluate the models
5. Run "c-datasets-and-tables.ipynb" to generate final tables
6. Run "c-train-flex.py" and "c-train-freeze.py" with appropriate flags to train intervened models
7. Run "c-val-flex.py" and "c-val-freeze.py" with appropriate flags to evaluate intervened models
8. Run "c-train8.py" and "c-val8.py" with appropriate flags to train and evaluate a model with more blocks

## Note
Supplementary contains the final error rates of models in the "checkpoints" folder.

## Packages Used

1. Python 3.13
2. Numpy
3. SciPy
4. Matplotlib
5. Jupyter Notebook
6. PyTorch
7. TorchVision
8. timm
