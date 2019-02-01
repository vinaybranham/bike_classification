# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:13:13 2018

@author: vinay
"""

import cv2
import os
import numpy as np
import tensorflow as tf


## Reading the training images
cur = os.getcwd()

# Path to the training images directory
datadir ="bikes-resized"
# List all sub directories
subdir = os.listdir(datadir)

size = 0
# Get the count of the total number of images
for dir in subdir:
    images = os.listdir(datadir + '/' + dir)
    size += len(images)

dataset = []        # stores the images
temp_labels = np.ndarray((size,1)) #stores the labels of images in order w.r.t "dataset"

i = 0

# Iterate through the sub-directories
for dir in subdir:
    # Make list of names of images in each sub-directory
    images = os.listdir(datadir + '/' + dir)
    # Iterate through the list of images
    for image in images:
        # Read each image in the sub-directory
        img = cv2.imread(datadir + '/' + dir + '/' + image)
        # Add it to the our "dataset" list
        dataset.append(img)
        # Add the corresponding label associated with the image
        # MountainBike --> 0
        # RoadBike --> 1
        if dir == "mountain_bikes":
            temp_labels[i] = 0
        else:
            temp_labels[i] = 1
        i = i+1


# Convert the array of "temp_labels" to binary class matrix
# Likely to extend to more classes.
labels = tf.keras.utils.to_categorical(temp_labels,num_classes=2)


## Creating the model for classification
# Initialiize model to add linear stack of layers
model = tf.keras.Sequential()

# Adding first Convolution layer - Also the first layer of the model which takes the 'input_shape' parameter
# Value of the 'input_shape' is based on the images used for the training --> (height, width, depth)
# Gives out 16 filter outputs which are passed on to the "ReLU" activation layer
model.add(tf.keras.layers.Conv2D(filters= 16, kernel_size=[3,3], activation='relu', input_shape=(300,300,3)))

# Adding second Convolution layer -- 16 filter outputs followed by "ReLU" activation
model.add(tf.keras.layers.Conv2D(filters= 16, kernel_size=[3,3], activation='relu'))

# Adding third Convolution layer -- 32 filter outputs followed by "ReLU" activation
model.add(tf.keras.layers.Conv2D(filters= 32, kernel_size=[3,3], activation='relu'))

# Adding the MaxPool to perform maximum pooling
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Adding fourth Convolution layer -- 32 filter outputs followed by "ReLU" activation
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[3,3], activation='relu'))

# Adding fifth Convolution layer -- 32 filter outputs followed by "ReLU" activation
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[3,3], activation='relu'))

# Adding the MaxPool to perform maximum pooling
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Adding sixth Convolution layer -- 16 filter outputs followed by "ReLU" activation
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=[3,3], activation='relu'))

# Adding seventh Convolution layer -- 16 filter outputs followed by "ReLU" activation
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=[3,3], activation='relu'))

# Adding the MaxPool to perform maximum pooling
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Adding eigth Convolution layer -- 16 filter outputs followed by "ReLU" activation
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=[3,3], activation='relu'))

# Adding the MaxPool to perform maximum pooling
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Adding the Flatten layer to perform FLattening
model.add(tf.keras.layers.Flatten())

# Adding a fully connected layer with 32 outputs
model.add(tf.keras.layers.Dense(units=32, activation='relu'))

# Adding a fully connected layer with 16 outputs
model.add(tf.keras.layers.Dense(units=16, activation='relu'))

# Adding a fully connected layer with 16 outputs
model.add(tf.keras.layers.Dense(units=16, activation='relu'))

# Adding a dropout layer to dropout to prevent overfitting
model.add(tf.keras.layers.Dropout(0.5))

# Adding the final fully connected layer with softmax activation
# The output of this layer gives the probability classifying the mountain bikes and the road bikes
# Note: Softmax in case I want to add more classes. Don't want to right now.
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))

# Configuring the model for training
# Using Adam optimizer with Categorical-crossentropy as loss and Accuracy as metrics
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])


## Training the model
traindataArr = np.array(dataset)

# Training the model with the input images
model.fit(traindataArr.reshape(traindataArr.shape[0], 300, 300, 3), labels, batch_size = 32, epochs=10)


## Testing the model
# Path to the test images directory
testdatadir ="test"
# Making a list of the names of test images
timages = os.listdir(testdatadir)

j=0

# Initializing the list to hold the test images
testdataset = []

# Reading the images from the directory
for image in timages:
    # Reading the images from the directory
    img = cv2.imread(testdatadir + '/' + image)
    testdataset.append(img)
    j=j+1

imgArr = np.array(testdataset)

# Testing the model by feeding the test images for classification
imgpredict = model.predict(imgArr.reshape(imgArr.shape[0], 300, 300, 3))
print(imgpredict)

# Displaying the images one by one with proper labelling after classification and the confidence
#  Use "Space bar" to iterate through the images
for k in range(len(testdataset)):
    imgArr = np.array(testdataset[k])
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    if np.argmax(imgpredict[k])==0:
        cv2.putText(imgArr,"mountain bike: "+str(imgpredict[k][0]*100),(10,10), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0))
    else:
        cv2.putText(imgArr,"road bike: "+str(imgpredict[k][1]*100),(10,10), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0))
    cv2.imshow('image',imgArr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
