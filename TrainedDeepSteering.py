
# coding: utf-8

# floyd run --gpu --mode jupyter --data itimmis/datasets/steering_new:steering_new

# In[1]:


# For deep learning
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.utils import np_utils
import tensorflow as tf

# Synthetic data generation (image augmentation)
from keras.preprocessing.image import ImageDataGenerator

# For image manipulation
from sklearn.feature_extraction import image
import scipy.misc

# For math
import numpy as np
from numpy import random
import math

# For file manipulation
import os
import glob
import pickle

# For graphing
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# For reading images
from matplotlib.image import imread

# For data manipulation
from sklearn.model_selection import train_test_split

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def radians_to_degrees(x):
    deg = math.degrees(x + math.pi)
    return deg


# In[3]:


def degrees_to_radians(x):
    rad = math.radians(x) - math.pi
    return rad


# In[4]:


def load_data_OLD():

    images = []
    labels = []

    cloud = True

    folder_names = ["campus_run1", "campus_run2", "run1", "run2", "run3", "second_spot1", "second_spot2"]

    if cloud == True:
        path = "/steering_new/"
    else:
        path = "/Users/itimmis/Desktop/ACTor/Data/SteeringNew/"

    # Loop folders
    for folder in folder_names:

        # open folder
        os.chdir(path + folder)

        # Loop files
        for file in glob.glob("*.jpg"):

            # Load image
            images.append(imread(file))

            # Extract label
            value = file.split('_')[-1].replace('.jpg','') # Remove the idx from filepath
            labels.append(radians_to_degrees(float(value)))

    # Convert to numpy arrays
    x = np.array(images)
    y = np.array(labels)

    # Extract Test Set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Extract Validation Set
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)

    # Normalize images
    x_train = x_train / 255.
    x_val = x_val / 255.
    x_test = x_test / 255.

    # Create data dictionary to contain train,val,test sets
    data_dict = {
        "x_train":x_train,
        "y_train":y_train,
        "x_val":x_val,
        "y_val":y_val,
        "x_test":x_test,
        "y_test":y_test
    }

    return data_dict


# In[5]:


def load_data(file):

    #file = '/steering_new/second_spot1/101_1.07687.jpg'

    # Load image
    img = imread(file)

    # Extract label
    value = file.split('_')[-1].replace('.jpg','') # Remove the idx from filepath
    label = radians_to_degrees(float(value))

    return img, label


# In[6]:


def init_model(img, hyperparameters):

    base_model = InceptionV3(weights=None, include_top=False, input_shape=img[0].shape)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # add a fully-connected layer
    x = Dense(hyperparameters["fc_size"], activation=hyperparameters["fc_activation"])(x)

    # add a logistic layer
    predictions = Dense(1, kernel_initializer='normal')(x)

    # train this model
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss=hyperparameters["loss"], metrics=hyperparameters["metrics"])

    return model


# In[7]:


def main():
    # Hyperparameters
    hyperparameters = {
        "batchsize" : 32,
        "fc_size" : 1024,
        "fc_activation" : 'relu',
        "epoch_finetune" : 30,
        "epoch_transfer" : 30,
        "loss" : "mean_squared_error",
        "metrics" : None,#["accuracy"]
        "monitor" : 'val_loss'
    }

    # # Load steering images

    fname = input("Enter filename to image: ")
    img_o, label_o = load_data(fname)

    img = np.array([img_o])
    img = img / 255.
    label = np.array([label_o]).reshape([1,1])

    print("img ",img.shape)
    print("lab ",label.shape)

    # # Create model
    model = init_model(img, hyperparameters)

    final_weights_path = "/Users/itimmis/Documents/AI/DeepSteeringWeights/final_weights.h5" #"/final_weights/final_weights.h5"

    model.load_weights(final_weights_path)

    prediction = model.predict(img,batch_size=1)

    print("Prediction: ", prediction[0][0])
    print("Actual: ", label[0][0])
    print("-------------------------")
    print("Error: ", abs(prediction[0][0] - label[0][0]))

main()
