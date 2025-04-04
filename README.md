# Hemorrhage detection using CNN and Transfer Learning

## Project Overview
This work presents a CNN-based binary classification model to detect hemorrhage in brain CT images. The model is trained on Dataset1 and evaluated on both Dataset1 and Dataset2, followed by fine-tuning experiments to assess cross-dataset performance.

## Motivation
This project was done as a part of the 'AI in Medical Image Analysis (DS261)' course at the Indian Institute of Science.

## Key Features

+ Binary classification of hemorrhage presence/absence
+ Cross-dataset evaluation
+ Transfer learning and fine-tuning experiments
+ Performance metrics analysis (precision, recall, f1-score)



## Datasets used
![image](https://github.com/user-attachments/assets/98e4944e-f148-4635-9c37-ac600e826e3c)

The datasets were provided by the course coordinator as part of the course. All the datasets can be found in this [repo](https://github.com/Saumya4321/DL261-class-Datasets). For this project, only Dataset1.zip and Dataset2.zip are used. Dataset1.zip is primarily a brain hemorrhage segmentation dataset with brain CT images and its corresponding segmentation mask for hemorrhage. While building the model, if the mask has any non-zero pixel, that particular scan is considered to have hemorrhage present. In contrast to this, Dataset2.zip has brain CT scans with a corresponding CSV file labelling each image into class '1' (has hemorrhage) and class '0' (no hemorrhage). 

## Experiments conducted

+ Initial training on Dataset-1 and evaluation on Dataset-1
+ Evaluation of model (trained on Dataset-1) on Dataset-2
+ Fine-tuning the model for Dataset-2
+ Evaluation of fine-tuned model on original dataset (Dataset-1)

## Data Preprocessing
### Normalization
After loading the training data from Dataset 1, normalization is applied to both the images
and their masks. This helps the features scale similarly and speeds up the training process.
### Converting and Loading Grayscale Images
All data is explicitly loaded as grayscale images to minimize the number of channels in the
loaded data.
### Data Augmentation
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
that the train images of dataset 1 look similar to the test ones of dataset 2 (domain-shift is small). This shows that the model has good generalization capability.


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
dataset-2 and though most images of dataset-1 and 2 are similar, dataset-2 has a few scan images which look different from dataset-1 as shown in Fig-3 (difference in data distribution - *domain shift*). Even though the domain shift is small, the model may overwrite some of the feature representations it originally learned from Dataset-1 to better fit Dataset-2. Hence, when the model is fine-tuned to adapt to this new data distribution, it reduces its ability to recognize patterns from the original dataset.
<br>
<br>
![image](https://github.com/user-attachments/assets/bd765ae0-1909-4edb-a352-23b69a0250b3)
<p align=center><em>Fig 3: Anomalous data found in Dataset-2 which is not there is Dataset-1</em></p>
<br>
<br>
This is called as <em>Catastrophic Forgetting</em>. Consequently, the accuracy of the model over the test data from
dataset 1 is lower.


![image](https://github.com/user-attachments/assets/f5906bff-7788-4d31-8be2-b35dd56ac253)

dataset 1

![image](https://github.com/user-attachments/assets/9a62e7d3-9d22-4b3f-84ca-1b86096527e9)

dataset 2 
![image](https://github.com/user-attachments/assets/4faa64d9-8ba2-497c-8918-ce00057730e6)

fine-tuning
![image](https://github.com/user-attachments/assets/3cffb38e-6211-4769-a942-ce4bf275fd02)

dataset 2


dataset 1



