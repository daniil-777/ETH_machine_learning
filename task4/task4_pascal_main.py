from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input #, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
import numpy as np

import task4_pascal_io as io4

def construct_preprocessing_model():
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
	inp = keras.Input(shape=input_shape)
	branch_inner = keras.layers.Dense(32, name='branch_inner', activation='relu')(inp)
	branch_output = keras.layers.Dense(16, name='branch_out')(branch_inner)
	return Model(
		inputs = inp,
		outputs = branch_output,
		name = 'siamese_branch',
	)


def construct_siamese_model():
	input_shape = (2048,) # preprocessing_model.output.shape

	input_a = keras.Input(shape=input_shape, name='input_a')
	input_b = keras.Input(shape=input_shape, name='input_b')
	input_c = keras.Input(shape=input_shape, name='input_c')

	# the weights are shared in all three branches
	branch = construct_branch_layers(input_shape)
	keras.utils.plot_model(branch, "siamese_branch.png", show_shapes=True)
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

	return Model(
		inputs = [input_a, input_b, input_c],
		outputs = comparison
	)

def assemble_triplets(data, triplets):
	a = np.array([data[t[0]] for t in triplets])
	b = np.array([data[t[1]] for t in triplets])
	c = np.array([data[t[2]] for t in triplets])
	return [a,b,c]

# images = io4.get_all_images()
# preprocessing_model = construct_preprocessing_model()
# imagenet_representations = preprocessing_model.predict(images)
# np.save('latent.npy', imagenet_representations)

print("\n\nload data")
imagenet_representations = np.load('latent.npy')
ir = imagenet_representations

test_triplets = io4.get_triplets('test_triplets.txt')
train_triplets = io4.get_triplets('train_triplets.txt')

print("\n\nconstruct model")
model = construct_siamese_model()
keras.utils.plot_model(model, "model.png", show_shapes=True)
model.summary()

# model.compile(
#     optimizer=keras.optimizers.Adam(1e-3),
#     loss=keras.losses.MeanAbsoluteError(),
# )

# print("\n\nassemble triplets")
# x = assemble_triplets(imagenet_representations, train_triplets)
# y = np.zeros(shape=(train_triplets.shape[0], 1))
# assert(len(x) == 3)
# assert(x[0].shape[0] == y.shape[0])

# print("\n\nfit model")
# model.fit(
#     x=x,
#     y=y,
#     batch_size=32,
#     epochs=3,
#     verbose=1,
#     callbacks=None,
#     validation_split=0.1,
#     shuffle=True,
# )

# dist_model = Model(
# 	inputs = model.input,
# 	outputs = [model.get_layer('dist_ab').output, model.get_layer('dist_ac').output]
# )

# print("\n\nevaluate model")
# [ab, ac] = dist_model(assemble_triplets(imagenet_representations, test_triplets))

# Task output specification
# 0 if the dish in A is closer in taste to C than to B
# 1 if the dish in A is closer in taste to B than to C

# output = (np.sign(ac-ab) + 1) / 2
# np.savetxt('out.csv', output.astype(int), fmt='%i')
