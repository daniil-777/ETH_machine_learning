from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input #, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd

data_path = './data/'
nr_images = 10000

def get_triplets(filename):
	path = data_path + filename
	data = pd.read_csv(path, sep=' ', header=None).to_numpy()
	assert(data.ndim == 2)
	assert(data.shape[1] == 3)
	return data

def get_image(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

def get_image_nr(nr):
	img_path = data_path + "food/{:05d}.jpg".format(nr)
	return get_image(img_path)

def get_all_images():
	return np.vstack([get_image_nr(idx) for idx in range(nr_images)])
