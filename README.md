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

The dataset folder contains the initial dataset with images of fields and roads. The data folder contains an extended dataset with more images of fields  for further fine-tuning.

- The fields class in the 'dataset' contains 45 images.
- The roads class in the 'dataset' contains 108 images.
- 
# Model Architecture and Hyperparameters
## Training

### Model Trained on `dataset`

The model trained on the `dataset` is stored in the `model_final.pth` file. This model was trained using a ResNet101 architecture. The training script and code used for this model can be found in the file "test_technical.ipynb" first part. It was trained using a ResNet101 architecture with the following hyperparameters:
- Architecture: ResNet101
- Learning Rate: 0.0001
- Batch Size: 16
- Number of Epochs: 10
- Loss Function: L1 Loss
  
- The choice of ResNet101 architecture was based on its deep and complex structure, allowing it to capture intricate features in images. The L1 loss was used to train the model to minimize the absolute differences between predicted and actual values.

### Model Trained on `data`

The model trained on the extended `data` dataset is stored in the `model_dataset_datafinal.pth` file. This model was also trained using a ResNet101 architecture and fine-tuned on the additional data the training script and code used for this model can be foun in the file "test_technical.ipynb" second part. This model was fine-tuned using the ResNet101 architecture with the following hyperparameters:

- Architecture: ResNet101
- Learning Rate: 0.00001
- Batch Size: 16
- Number of Epochs: 7
- Loss Function: L1 Loss
- The choice of fine-tuning the ResNet101 architecture allowed the model to adapt and specialize for improved accuracy on the extended dataset.

# Data Preprocessing
For both models, the following image transformations were applied to the input images:

- Resize to 224x224 pixels
- Random horizontal flip with a probability of 50%
- Random rotation by up to 5 degrees
- Convert to tensor
- Normalize pixel values to a range of (-1, 1)

## Inference

To predict the class of a single image using either of the trained models, you can use the `predict_image_class` function provided in the 'test_technical.ipynb' first part for `model_final.pth` and `predict_image_class1` for  `model_dataset_datafinal.pth`.

# Results
Both models achieved high accuracy on separate test sets, effectively distinguishing between images of fields and roads:

## Model Trained on dataset
- Test Loss: 0.0152
- Test Accuracy: 100.00%
## Model Trained on data
- Test Loss: 0.0070
- Test Accuracy: 100.00%

## Additional Note
It's worth mentioning that during the course of the project, I discovered two images initially categorized as "roads" in the fields class of the dataset. These images were subsequently removed from the fields class and utilized for prediction purposes.

For image predictions it is better to use `model_dataset_datafinal.pth`.

## Future Improvements
- Explore alternative architectures and hyperparameters to further improve performance.
- Gather and label a more diverse dataset to enhance model generalization.
- Experiment with additional data augmentation techniques.
  
