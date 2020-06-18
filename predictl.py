import cv2
import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import csv

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

test_csv = "./data/test.csv"		# Path to test.csv file
test_images = "./data/test/"		# Path to test images folder

model = tf.keras.models.load_model('./cnn.model')

# Preprocess image before testing
def preprocess(image):
    IMG_SIZE = 200
    img_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img_array = img_array / 255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def img_predict(image):
	"""
	Function takes a path of images and return 
	the predicted label.
	"""

	img = preprocess(image)
	predict = model.predict(img)
	label = np.argmax([predict[0]])
	return label

def prediction(path, test_images):
	"""
	Function takes the path of the csv file, and the path to images folder
	Extracts the test image using csv from the  test folder
	Passes it to predict and appends the res to test_result.csv
	"""
	csv_file = "predictions.csv"
	res = []
	df = pd.read_csv(path)
	images = df['Image']
	images = df.to_numpy(images)		# Convert df to numpy
	print(f"Test Size: {len(images)}")
	for img in images:
		print(f"Current image {img}")
		img = img[0]
		img_path = os.path.join(test_images, img)
		label = img_predict(img_path)
		res.append(CATEGORIES[label])

	print("Prediction complete . . .")
	print("Writing CSV file ")

	with open(csv_file, 'w') as f:
		w = csv.writer(f)
		w.writerow(["Image", "target"])
		for i in range(len(images)):
			w.writerow([images[i][0], res[i]])



prediction(test_csv, test_images)