# Hemorrhage detection using CNN and Transfer Learning

## Project Overview
This project implements a deep learning-based binary classification model to detect the presence of hemorrhage in medical images. The model is trained on Dataset1 and evaluated on both Dataset1 and Dataset2, followed by fine-tuning experiments to assess cross-dataset performance.

## Motivation
This project was done as a part of my DS261 assignment.

## Key Features

+ Binary classification of hemorrhage presence/absence
+ Cross-dataset evaluation
+ Transfer learning and fine-tuning experiments
+ Performance metrics analysis (precision, recall, f1-score)

## Datasets used

## Data Preprocessing
#### Normalization
After loading the training data from Dataset 1, normalization is applied to both the images
and their masks. This helps the features scale similarly and speeds up the training process.
#### Converting and Loading Grayscale Images
All data is explicitly loaded as grayscale images to minimize the number of channels in the
loaded data.
#### Data Augmentation
Since the provided dataset contains a small number of scans (around 400), data augmentation
techniques are applied to help the model learn the classes better. The following data
augmentation transformations are applied to the training data from Dataset 1:
+ Rotation
+ Zooming in and out
+ Horizontal flip
Data augmentation is applied only to the training data, not to the validation and test data.

## Model Architecture
The model architecture consists of 5 repetitions of Conv2d, max pooling layers, and dropout
layers. This is followed by a flattening layer and two dense layers. The final dense layer gives
two outputs corresponding to the two classes ”Hemorrhage” and ”No Hemorrhage,” using
the softmax activation function.
<br>
Since the dataset is quite small, L2 regularization is applied in each of the convolutional
layers to help the model generalize better. In addition to this, dropout layers are also used
for ensuring better generalization.
<br>
As the task is binary classification, the Binary Cross Entropy loss function was chosen, as
it is shown to be best suited for the given task. The Adam optimizer with a small learning
rate of 1e-4 is chosen so that the model can learn intricate features.In order to increase
the chances of finding the global minima for loss, a custom cosine annealing learning rate
scheduler is applied. This varies the learning rate (after initial warmup of 7 epochs) according
to the Cosine function. It is seen that this has helped the model to reach a good validation
accuracy.
<br>
In order to further decrease overfitting, early stopping with a patience of 7 epochs is
applied. A learning rate callback function is also defined such that it automatically reduces
the learning rate if validation loss hits a plateau. This is done to better find the minimal
validation loss point.
<br>
The model was set to train over the training data from dataset-1 for 40 epochs. Early
stopping algorithm caused the training to stop after 18 epochs.
