#!/usr/bin/python3

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
import numpy as np
from pathlib import Path

import task4_io as io4

def construct_imagenet_model():
	pretrained_model = ResNet50(weights='imagenet')
	inner_layer = pretrained_model.get_layer(name='avg_pool')
	return Model(
		inputs = pretrained_model.input,
		outputs = inner_layer.output
	)

# Takes two shapes that are assumed to be equal and
# returns a shape with the height of the input and
# the width of 1.
def lambda_output_shape(shapes):
	return (shapes[0][0], 1)

# Taken from https://keras.io/examples/mnist_siamese/
def euclidean_distance_lambda(vects):
	x, y = vects
	sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sum_square, K.epsilon()))

def distance_comparison_lambda(inputs):
	ab, ac = inputs
	return K.clip(ab/(ac + K.epsilon()) - 0.6, min_value=0, max_value=None) # PARAM:

# 0.45 : 0.68412372522
# 0.60 : 0.687590454714
# 0.68 : 0.684460300899
# 0.80 : 0.674564975935

def construct_distance_layer(name):
	return keras.layers.Lambda(
		euclidean_distance_lambda,
		output_shape=lambda_output_shape,
		name=name
	)

def construct_branch_layers(input_shape):
	dropout_rate = 0.0 # PARAM: DISABLE NODE PROBABILITY

	# PARAM: try add more layers but change the number 6 below in create_and_train_model, 2 per Dense, A and b
	inp = keras.Input(shape=input_shape)
	branch_inner = keras.layers.Dense(128, name='branch_inner_1', activation='relu', kernel_regularizer=keras.regularizers.l1(0.001))(inp) # PARAM: maybe l1 to l2 or l1_l2 (-> keras docs)
	branch_inner = keras.layers.Dropout(dropout_rate, name='branch_dropout_1')(branch_inner)
	branch_inner = keras.layers.Dense(64, name='branch_inner_2', activation='relu')(branch_inner) # PARAM: maybe regularizer here also?
	branch_inner = keras.layers.Dropout(dropout_rate, name='branch_dropout_2')(branch_inner)
	branch_output = keras.layers.Dense(32, name='branch_out')(branch_inner)
	return Model(
		inputs = inp,
		outputs = branch_output,
		name = 'siamese_branch',
	)


def construct_siamese_model(size_tuple):
	input_shape = size_tuple[1:]

	input_a = keras.Input(shape=input_shape, name='input_a')
	input_b = keras.Input(shape=input_shape, name='input_b')
	input_c = keras.Input(shape=input_shape, name='input_c')

	# the weights are shared in all three branches
	branch = construct_branch_layers(input_shape)
	# keras.utils.plot_model(branch, "siamese_branch.png", show_shapes=True)
	out_a = branch(input_a)
	out_b = branch(input_b)
	out_c = branch(input_c)

	dist_ab = construct_distance_layer("dist_ab")([out_a, out_b])
	dist_ac = construct_distance_layer("dist_ac")([out_a, out_c])

	comparison = keras.layers.Lambda(
		distance_comparison_lambda,
		output_shape=lambda_output_shape,
		name="distance_comparison"
	)([dist_ab, dist_ac])

	model = Model(
		inputs = [input_a, input_b, input_c],
		outputs = comparison
	)
	# keras.utils.plot_model(model, "model.png", show_shapes=True)
	return model

def assemble_triplets(data, data_flipped, triplets, use_flipped):
	d = data
	df = data_flipped

	# This code looks way scarier than it should...
	# combinations is all 8 elements of the cartesian product of three bools, it's used
	# to select whether to use a normal or a flipped image for A,B,C.
	# After that we just return those 8 rows per triplet
	# If use_flipped is false then just 1 row is returned per triplet

	combinations = [[0,0,0]]
	if use_flipped:
		combinations = [[a,b,c] for a in [0,1] for b in [0,1] for c in [0,1]]

	return [np.array([[d,df][c[i]][t[i]] for t in triplets for c in combinations]) for i in [0,1,2]]


def apply_imagenet_to_dataset():
	imagenet_model = construct_imagenet_model()

	images = io4.get_all_images()
	out = imagenet_model.predict(images)
	np.save(io4.data_processed_path + 'imagenet_food.npy', out)

	# images_flipped = io4.get_all_images_flipped()
	# out_flipped = imagenet_model.predict(images_flipped)
	# np.save(io4.data_processed_path + 'imagenet_food_flipped.npy', out_flipped)

def create_and_train_model(food_data, food_data_flipped, train_triplets, use_flipped):
	model = construct_siamese_model(np.shape(food_data))
	model.compile(
		optimizer=keras.optimizers.Adam(1e-3),
		loss=keras.losses.MeanAbsoluteError(),
	)

	x = assemble_triplets(food_data, food_data_flipped, train_triplets, use_flipped)
	y = np.zeros(shape=(x[0].shape[0], 1))

	# It can take a while to train the model, so we save the weights and can load
	# them again later.
	load_saved_weights = False
	weights_dir = 'weights/'

	if load_saved_weights:
		saved_weights = [np.load(weights_dir + 'w{}.npy'.format(i)) for i in range(6)]
		model.set_weights(saved_weights)
	else:
		model.fit(
			x=x,
			y=y,
			batch_size=1024, # PARAM: INCREASE FOR MORE ACCURATE GD
			epochs=30, # PARAM: NUMBER OF DATA INPUT PASSES
			verbose=1,
			callbacks=None,
			validation_split=0.1,
			shuffle=True,
		)

		Path(weights_dir).mkdir(parents=True, exist_ok=True)
		w = model.get_weights()
		for i in range(6):
			np.save(weights_dir + 'w{}.npy'.format(i),w[i])

	return model


def main():
	imagenet_food = np.load(io4.data_processed_path + 'imagenet_food.npy')
	torch_food = np.load(io4.data_processed_path + 'torch_food.npy')
	all_food_data = np.concatenate((imagenet_food, torch_food), axis=1)

	train_triplets = io4.get_triplets('train_triplets.txt')
	test_triplets = io4.get_triplets('test_triplets.txt')

	model = create_and_train_model(all_food_data, None, train_triplets, use_flipped=False)

	dist_model = Model(
		inputs = model.input,
		outputs = [model.get_layer('dist_ab').output, model.get_layer('dist_ac').output]
	)

	[ab, ac] = dist_model(
		assemble_triplets(all_food_data, None, test_triplets, use_flipped=False),
		training=False
	)

	# Task output specification
	# 0 if the dish in A is closer in taste to C than to B
	# 1 if the dish in A is closer in taste to B than to C

	output = (np.sign(ac-ab) + 1) / 2
	np.savetxt('out.csv', output.astype(int), fmt='%i')

	print("\n\ndone")

if __name__ == '__main__':
	# generate the files imagenet_food.npy
	apply_imagenet_to_dataset()

	# train and evaluate the model
	main()
