#!/usr/bin/python3.6

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import random

train_path = "./data/train/"
label_path = "./data/train.csv"
images = os.listdir(train_path)        # List of image names
IMG_SIZE = 200
CATEGORIES = [
    "manipuri", 
    "bharatanatyam", 
    "odissi", 
    "kathakali", 
    "kathak", 
    "sattriya", 
    "kuchipudi",
    "mohiniyattam"
]


def read_csv(train_csv):
    """
    The function reads the label csv file and returns
    two arrays: image_name, and image_labels
    image_name -> store the names of imagefiles
    image_label -> stores the label/category for the image at same index 
        in image_name
    """
    df = pd.read_csv(train_csv)
    image_names = df['Image'].to_numpy()
    image_labels = df['target'].to_numpy()
    print(df.head())
    return image_names, image_labels

train_images, train_labels = read_csv(label_path)

def create_training_data(images, image_names, image_labels):
    """
    Function takes a list of images to be trained.
    Returns a numpy array of processed image arrays
    """
    training_data = []
    count = 0
    print("Length of trainig data: ", len(images))
    for img in images:
        img_index = np.where(image_names == img)        # Get the name index of label.csv
        img_label = image_labels[img_index]             # Extract named label
        img_path = os.path.join(train_path, img)
        img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))        # Reshape image
        label = CATEGORIES.index(img_label[0])
        training_data.append([img_arr, label])
        count += 1
    return training_data

# preprocess trainig data
train_data = create_training_data(images, train_images, train_labels)


# Save the training data to .npy file
def save_data(train_data):
    print("saving data")
    random.shuffle(train_data)    # shuffle training data
    X = []
    y = []
    for features, label in train_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    np.save('features.npy', X)
    np.save('labels.npy', y)
    print("Features and labels saved on disk . . . ")

save_data(train_data)
