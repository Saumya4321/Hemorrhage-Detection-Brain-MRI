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
<br>
Since the dataset is quite small, L2 regularization is applied in each of the convolutional
layers to help the model generalize better. In addition to this, dropout layers are also used
for ensuring better generalization.
<br>
<br>
As the task is binary classification, the Binary Cross Entropy loss function was chosen, as
it is shown to be best suited for the given task. The Adam optimizer with a small learning
rate of 1e-4 is chosen so that the model can learn intricate features. In order to increase
the chances of finding the global minima for loss, a custom cosine annealing learning rate
scheduler is applied. This varies the learning rate (after initial warmup of 7 epochs) according
to the Cosine function. It is seen that this has helped the model to reach a good validation
accuracy.
<br>
<br>
In order to further decrease overfitting, early stopping with a patience of 7 epochs is
applied. A learning rate callback function is also defined such that it automatically reduces
the learning rate if validation loss hits a plateau. This is done to better find the minimal
validation loss point.
<br>
<br>
The model was set to train over the training data from dataset-1 for 40 epochs. Early
stopping algorithm caused the training to stop after 18 epochs.
<br>
<br>
![image](https://github.com/user-attachments/assets/8cd235e6-e9e8-4484-bdee-6975a6c5857e)
<br>
<br>
Next, the model is tested on the test data of dataset-1. A test loss of 0.8120 and test
accuracy of 0.78 was obtained. Following is a tabulation of the results obtained
<br>
<br>
![image](https://github.com/user-attachments/assets/ba1732aa-a71a-45cc-a9c1-8f2330458f14)
<br>
<br>
Here, recall refers to True Positive Rate (TPR) and f1 score refers to the dice score.

## Testing the same model on test data of Dataset-2
Test data of dataset-2 was loaded,
the scan images were normalized and one-hot encoding was performed on the labels. On
evaluation of the model on this data, a test loss of 0.8981 and a test accuracy of 0.6167 was
obtained. The precision, recall and dice scores were as follows -
<br>
<br>
![image](https://github.com/user-attachments/assets/79414028-55ac-40a3-9183-5c17dacc62cc)
<br>
<br>
As seen above, the model’s performance seems similar. This can be attributed to the fact
that the train images of dataset 1 look similar to the test ones of dataset 2.


## Fine Tuning the model over Dataset 2
The same model is taken and only the initial learning rate of the adam optimizer is changed,
in order for the model to better learn the intricacies of dataset 2. The initial learning rate
of adam optimizer is set to 3e-4.
<br>
<br>
![image](https://github.com/user-attachments/assets/22afc6ae-a7db-49f1-a4ed-49fc7eb33fea)
<br>
<br>
When the model was evaluated on test data from dataset-2, a test loss and test accuracy
of 0.7421 and 0.7167 was reported. A higher accuracy score is seen in this case than when
compared to the previous case. This is due to the fact that the model was fine tuned
specifically to be better suited to the training images of dataset 2.
The detailed classification report is given below
<br>
<br>
![image](https://github.com/user-attachments/assets/6443a934-e8bb-4ad5-a268-47fb705bfd43)
<br>
<br>
Now, this fine-tuned model is evaluated on test data from Dataset-1 also. The test loss
and test accuracy are 0.8861 and 0.60. The detailed classification report is given below -
<br>
<br>
![image](https://github.com/user-attachments/assets/58d66790-c323-4b4b-9933-e24e08a7338d)
<br>
<br>
This is lower than the first case. This is because the model is specifically fine-tuned for
dataset-2 and though most images of dataset-1 and 2 are similar, dataset-2 has the same
additional (faulty) variety of images in it, as shown below. This isn’t present in dataset-1 at
all.
<br>
<br>
![image](https://github.com/user-attachments/assets/bd765ae0-1909-4edb-a352-23b69a0250b3)
<br>
<br>
Hence, as the model is fine-tuned to recognise the patterns in the above varied instances
of data, which isn’t present in dataset 1, the accuracy of the model over the test data from
dataset 1 is lower.






