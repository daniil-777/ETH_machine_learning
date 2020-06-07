from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input #, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
import numpy as np
from pathlib import Path


import task4_pascal_io as io4

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
	return K.clip(ab/(ac + K.epsilon()) - 0.8, min_value=0, max_value=None)

def construct_distance_layer(name):
	return keras.layers.Lambda(
		euclidean_distance_lambda,
		output_shape=lambda_output_shape,
		name=name
	)

def construct_branch_layers(input_shape):
	dropout_rate = 0.2

	inp = keras.Input(shape=input_shape)
	branch_inner = keras.layers.Dense(64, name='branch_inner_1', activation='relu')(inp)
	branch_inner = keras.layers.Dropout(dropout_rate, name='branch_dropout_1')(branch_inner)
	branch_inner = keras.layers.Dense(32, name='branch_inner_2', activation='relu')(branch_inner)
	branch_inner = keras.layers.Dropout(dropout_rate, name='branch_dropout_2')(branch_inner)
	branch_output = keras.layers.Dense(16, name='branch_out')(branch_inner)
	return Model(
		inputs = inp,
		outputs = branch_output,
		name = 'siamese_branch',
	)


def construct_siamese_model():
	input_shape = (2048,)

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

def assemble_triplets(data, data_flipped, triplets, use_flipped=False):
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
	np.save('imagenet.npy', out)

	images_flipped = io4.get_all_images_flipped()
	out_flipped = imagenet_model.predict(images_flipped)
	np.save('imagenet_flipped.npy', out_flipped)

def create_and_train_model(imagenet_rep, imagenet_rep_flipped, train_triplets):
	model = construct_siamese_model()
	model.compile(
		optimizer=keras.optimizers.Adam(1e-3),
		loss=keras.losses.MeanAbsoluteError(),
	)

	x = assemble_triplets(imagenet_rep, imagenet_rep_flipped, train_triplets, use_flipped=True)
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
			batch_size=32,
			epochs=5,
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
	imagenet_rep = np.load('imagenet.npy')
	imagenet_rep_flipped = np.load('imagenet_flipped.npy')

	train_triplets = io4.get_triplets('train_triplets.txt')
	test_triplets = io4.get_triplets('test_triplets.txt')

	model = create_and_train_model(imagenet_rep, imagenet_rep_flipped, train_triplets)

	dist_model = Model(
		inputs = model.input,
		outputs = [model.get_layer('dist_ab').output, model.get_layer('dist_ac').output]
	)

	[ab, ac] = dist_model(
		assemble_triplets(imagenet_rep, imagenet_rep_flipped, test_triplets, use_flipped=False),
		training=False
	)

	# Task output specification
	# 0 if the dish in A is closer in taste to C than to B
	# 1 if the dish in A is closer in taste to B than to C

	output = (np.sign(ac-ab) + 1) / 2
	np.savetxt('out.csv', output.astype(int), fmt='%i')

	print("\n\ndone")




################################################################################
################################################################################
#######
#######   change stuff here:
#######

# Uncommet this function to generate the files imagenet.npy and imagenet_flipped.npy
apply_imagenet_to_dataset()

# Uncomment this function once the files imagenet.npy and imagenet_flipped.npy are created
# to train and evaluate the model
main()
