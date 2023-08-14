# Trimble-test
Classifier  Field &amp; Road  using Deep learning
# Field vs Road Classifier

This repository contains code for training a classifier to distinguish between images of fields and roads using deep learning. Two different models were trained: one on the `dataset` and another on the `data` folder.

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- PIL

You can install the required packages using the following command:
pip install torch torchvision matplotlib Pillow

## Dataset

The `dataset` folder contains the initial dataset with images of fields and roads. The `data` folder contains an extended dataset with more images of fields and roads for further fine-tuning.

## Training

### Model Trained on `dataset`

The model trained on the `dataset` is stored in the `model_final.pth` file. This model was trained using a ResNet101 architecture. The training script and code used for this model can be found in the repository "test_technical.ipynb".
