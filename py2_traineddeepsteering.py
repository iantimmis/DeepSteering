# For deep learning
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.utils import np_utils
import tensorflow as tf

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

def radians_to_degrees(x):
    deg = math.degrees(x + math.pi)
    return deg

def degrees_to_radians(x):
    rad = math.radians(x) - math.pi
    return rad

#def load_data(file):
def load_data():
    
    file = '/steering_new/second_spot1/101_1.07687.jpg'

    # Load image
    img = imread(file)

    # Extract label
    value = file.split('_')[-1].replace('.jpg','') # Remove the idx from filepath
    label = radians_to_degrees(float(value))

    return img, label

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

    # fname = input("Enter filename to image: ")

#img_o, label_o = load_data(fname)
    img_o, label_o = load_data()

    img = np.array([img_o])
    img = img / 255.
    label = np.array([label_o]).reshape([1,1])

    print "img ",img.shape
    print "lab ",label.shape

    # # Create model
    model = init_model(img, hyperparameters)

    final_weights_path = "/weights/final_weights.h5"

    model.load_weights(final_weights_path)

    prediction = model.predict(img,batch_size=1)

    print "Prediction: ", prediction[0][0]
    print "Actual: ", label[0][0]
    print "-------------------------"
    print "Error: ", abs(prediction[0][0] - label[0][0])

main()
