# Automated Pneumonia Diagnosis Using Deep Learning

# Introduction

## Problem 

Chest X-rays are primarily used for the diagnosis of pneumonia. However, even for a trained radiologist, it is a challenging task to examine chest X-rays. To solve this problem, deep learning (DL), a branch of machine learning (ML), are developed to detect hidden features in images which are not apparent or cannot be detected even by medical experts. With AI system aiding medical experts in expediting the diagnosis, earlier treatment can be prescribed, resulting in improved clinical outcomes.

We are going to build an automated methods to detect and classify pneumonia & normal from medical X-ray images.

# Data

An input of total x-ray images of 5853 were downloaded from kaggle,

https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia.

The data images in kaggle was already split into train, validation and test data. 

### Train Data
3543 validated images filenames belonging to 2 classes.

### Validation Data
1162 validated images filenames belonging to 2 classes.

### Test Data 
1148 validated images filenames belonging to 2 classes.


The data folder can be found in this repository.
The colaboratory notebook can be found as "Pneumonia_Detection.ipynb" in Pneumonia_Diagnosis_Project folder or in the main branch in this repository.
The saved models can be found in "saved models" folder in Pneumonia_Diagnosis_Project folder.
Note that only that model 2, 6, 9, 12, 14, 15 are not saved and not present in saved models folder. These models did not performed good and therefore deemed not useful for future purposes.
A presentation with best results can be found in Pneumonia Detection by Deep Learning.pdf in this repository. 

# Methodology

We used Convolutional neural network (CNN) to build and automated method to detect pneumonia X-ray from nromal X-ray images. CNN is a powerful tool due to its ability to extract features and learn to distinguish between different classes such as positive and negative, infected and healthy, or in this case, Pneumonia and Normal.

For large datasets and deep networks, kernel regularization is a must. You can use either L1 or L2 regularization. If you detect signs of overfitting, consider using L2 regularization. Tune the amount of regularization, starting with values of 0.0001-0.001. For bias and activity, we recommend leaving at the default values for most scenarios

In Keras, you build a CNN architecture using the following process:

1) Reshape the input data into a format suitable for the convolutional layers.

2) For class-based classification, one-hot encode the categories using the to_categorical() function.

3) Build the model  by initializing a Sequential model. 

4) Add convolutional layers. Convolutional layers apply a filter to input and create a feature map that summarizes the presence of detected features in the input.  The input image gets smaller and smaller as it progresses through the network but it also gets deeper and deeper with feature map. I add one Conv2D layer with a filter/kernel size of 3 x 3, adding ReLU activation function to set negative values to zero, and specify input_shape = (150, 150, 3).

5) Add Pooling layer is sandwiched between two successive convolutional layers to reduce the spatial size of the convoluted feature/ parameters in the network. MaxPooling is the most common pooling methods to reduce image size by pooling the most important feature. Here I use MaxPool2D with a pool size of 2 or 3, meaning it divides each spatial dimension by a factor of 2 or 3.

6) Repeat the above one or more times.

7) Add a “flatten” layer which prepares a vector for the fully connected layers. Flatten layer is added to convert each input image into a 1D array: if it receives input data X, it computes X.reshape(-1, 1). The flatten data that comes out of the convolutions and is then fed to the fully connected layers.

8) Add one or more fully hidden dense connected layer. Typically you will follow each fully connected layer with a dropout layer.  

9) Add 1 output hidden Dense layer with one neuron using activation = 'sigmoid'.

10) Compile the model using model.compile()

11) Train the model using model.fit().

12) Use model.predict() to generate a prediction.

Sometimes, I add callbacks to monitor a specific parameter of the model, in this case, val_acc. Since I use a validation set during training, I set save_best_only = True in ModelCheckpoint to specify that the model will only be saved when its performance on the validation set is best.

I also set patience = 10 for EarlyStopping, meaning that the model will stop training if it doesn't see any improvement in val_acc in 10 epochs.


# Model Training

Training images and their corresponding true labels, used CNN deep learning and adjust weights(parameters) through hypertuning parameters such as: padding (valid or same),Dropout(0.2),with and without Data Augmentation techniques, Kernel regularizers and bias regularizers l1 or l2 or l1-l2, Batch normalization. We also optimized the number of convolutional layers (2-5), dense layers (1-4), number of filters, and number of dense layer neurons. Moreover, we also otimized the model on diffrent optimizers during model compiling (optimizer = Adam or optimizer = RMSprop)

Validation images and their corresponding true labels (we use these labels only to validate the model and not during the training phase)

 We ran all the models for 30 epochs.
 
## Model 0: Visualization of loss and accuracy of training and validation data
 
 ![image](https://user-images.githubusercontent.com/53411455/117719412-5f389600-b1ab-11eb-8484-ed4edf7cb94b.png)
 ![image](https://user-images.githubusercontent.com/53411455/117720463-abd0a100-b1ac-11eb-88d7-0230affdf263.png)

## Model 1: Visualization of loss and accuracy of training and validation data
  ![image](https://user-images.githubusercontent.com/53411455/117719618-9b6bf680-b1ab-11eb-9cd8-c529a8a26165.png)
  ![image](https://user-images.githubusercontent.com/53411455/117720494-b4c17280-b1ac-11eb-9b1b-44cee530f4e8.png)

  
# Conclusion

This study presents a deep CNN based approach for the automatic detection of Pneumonia vs Normal X-ray images.

We have demonstrated how to distinguish between Normal and Pneumonia Chest X-Rays with our model having an test accuracy of 89%-92% and a recall of 94%-99%.

The best model which gave promising results are: Model 0 , Model 4 & Model 11.

We constucted a Convolutional Neural Network model from scratch to extract features from a given Chest X-Ray image and classify it to determine if a person is infected with Pneumonia.

## Model 0: Visualization of Test Data Predictions

![image](https://user-images.githubusercontent.com/53411455/117719831-dff79200-b1ab-11eb-80c5-17cbad913c17.png)


## Model 11: Visualization of Test Data Predictions

![image](https://user-images.githubusercontent.com/53411455/117719875-f1d93500-b1ab-11eb-867e-30239eaebc4b.png)


# Recommendations

Incorporate our model to see how it works in hospitals so that it can assist health professionals diagnose patients with pneumonia. Ofcourse these reports have to be validated.  This needs to be certified by health professionals.
This model should be run under the supervision of a radiologist to enhance accuracy/recall to improve treatment outcomes which will increase hospitals' ratings and fundings.

This model can be incorporated with the software program of X-ray machines or MRI scans to detect automatically the results once the scan of the patient is over.
  

# Future Work
  
We need more data to run our validation set on so that we can be sure of the way the model is predicting. Also, work towards getting our Transfer Learning model to work better.

Build a multi-class classification deep learning model to distinguish between Normal, Viral Pneumonia, and Bacterial Pneumonia

Combine CNN models with other classifiers such as Support Vector Machine (SVM)

Tune parameters such as learning rate, batch size, try another optimizer, number of layers, different types of layer, number of neurons per layer, and the type of activation functions for each layer. GridSearchCV or RandomizedSearchSV for optimization issues.  
 

