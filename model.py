#!/usr/bin/python3.6

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from sklearn.model_selection import train_test_split
import time

NAME = "dance-form-classifier-{}".format(time.strftime("%d-%h:%m:%S"))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

classes = [ x for x in range(8)]

#Model name for tensor board
#Model name for tensor board
X = np.load('features.npy')
y = np.load('labels.npy')

X = X/255.0      # Normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print("Train images shape ", X.shape[1:])


ep = 20         # Epochs
bs = 32         # Batch size

# CNN Model
def cnn_model(X_train, X_test , y_train, y_test):

    model = Sequential()

    # Layer I
    model.add(Conv2D( 64, (3, 3), activation="relu", input_shape=X.shape[1:] ))
    model.add(MaxPooling2D(2,2))

    # Layer II
    model.add(Conv2D(128,(3,3),activation="relu"))
    model.add(MaxPooling2D(2,2))

    # Layer III
    model.add(Conv2D(128,(3,3),activation="relu"))
    model.add(MaxPooling2D(2,2))

    # Layer IV
    model.add(Conv2D(256,(3,3),activation="relu"))
    model.add(MaxPooling2D(2,2))
    
    # Layer VI
    model.add(Conv2D(64,(3,3),activation="relu"))
    model.add(MaxPooling2D(2,2))

    # Flatten the input
    model.add(Flatten())

    # Dense Layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation="softmax"))


    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=bs, epochs=ep, validation_data = (X_test, y_test), callbacks = [tensorboard])

cnn_model(X_train, X_test, y_train, y_test)
