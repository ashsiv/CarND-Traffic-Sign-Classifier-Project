## Project: Traffic Sign Classification

Overview
---
In this project, I have used deep neural networks and convolutional neural networks to classify traffic signs. 

you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. 

To run this project, use the "Traffic_Sign_Classifier.ipynb" notebook.

Project Goals
---
* Dataset summary and visualization.
* Design, train and test a Lenet model architecture.
* Use the model to make predictions on new images & analyze the softmax probabilities of the new images.


### Dataset summary and visualization

Here is the statistics of the dataset. A 80-20 train-validation split is taken.

* Number of training examples   = 31367(80.0%)
* Number of validation examples = 7842(20.0%)
* Number of testing examples    = 12630
* Input Image data shape        = (32, 32, 3)
* Number of classes             = 43

Plotting the distribution, we can observe that both in training and testing set, the % of images per class is roughly the same.
<p align="center">
    <img src="./Distribution.PNG" alt="Image" />
</p>
