#!/usr/bin/python3.6

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import regularizers, applications
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
import time

NAME = "dance-form-classifier-{}".format(time.strftime("%d%h-%m-%S"))

#Model name for tensor board
tensorboard = TensorBoard(log_dir="""logs\{}""".format(NAME))
CLASSES=8
classes = [ x for x in range(8)]


X = np.load('features.npy')
y = np.load('labels.npy')

X = X/255.0      # Normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print("Train images shape ", X.shape[1:])


ep = 20         # Epochs
bs = 32         # Batch size



# CNN Model
def cnn_model(X_train, X_test , y_train, y_test):

    base_model = applications.VGG16(include_top=False, input_shape=X_train.shape[1:], weights='imagenet',classes=CLASSES)

    # Freezing VGG16 layers
    for layer in base_model.layers:
        layer.trainable=False
    
    last_layer = 'block5_pool'
    model = Model(base_model.input, base_model.get_layer(last_layer).output)

    model.layers[-1].output.shape
    model = Sequential()

    model.add(base_model)      # Stack vgg16 

    model.add(Conv2D(128,(3,3),activation="relu", input_shape=model.layers[-1].output.shape, data_format='channels_first'))
    model.add(MaxPooling2D(2,2))

    # model.add(Conv2D(128,(3,3),activation="relu"))
    # model.add(MaxPooling2D(2,2))

    model.add(Flatten())        # Flatten the output

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(CLASSES, activation="sigmoid"))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=bs, epochs=ep, validation_data = (X_test, y_test),  callbacks=[tensorboard])

    # Save model
    model.save('cnn.model')


    model.summary()

cnn_model(X_train, X_test, y_train, y_test)

