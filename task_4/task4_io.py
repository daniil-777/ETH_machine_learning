from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input #, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd

data_path = './data/'
data_processed_path = './data_processed/'
nr_images = 10000

def get_triplets(filename):
	path = data_path + filename
	data = np.array(pd.read_csv(path, sep=' ', header=None).values)
	assert(data.ndim == 2)
	assert(data.shape[1] == 3)
	return data

def get_image(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	return image.img_to_array(img)

def get_image_nr(nr):
	img_path = data_path + "food/{:05d}.jpg".format(nr)
	return get_image(img_path)

def flip(img):
	return img[:,::-1]

def preprocess_image(img):
	x = np.expand_dims(img, axis=0)
	x = preprocess_input(x)
	return x

def get_all_images():
	return np.vstack([preprocess_image(get_image_nr(idx)) for idx in range(nr_images)])

def get_all_images_flipped():
	return np.vstack([preprocess_image(flip(get_image_nr(idx))) for idx in range(nr_images)])
