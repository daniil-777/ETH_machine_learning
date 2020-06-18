#!/usr/bin/python3

# tested with these library versions:
#     torch: 1.5.0
#     torchvision: 0.7.0a0+148bac2
#     PIL: 6.2.1

import torch
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image
import numpy as np
from pathlib import Path

# adapted from https://github.com/Prakhar998/Food-Classification
# Download food_classifier.pt from https://github.com/Prakhar998/Food-Classification/raw/master/food_classifier.pt

def get_model():
	model = models.densenet201(pretrained=True)
	model.classifier = nn.Sequential(nn.Linear(1920,1024),nn.LeakyReLU(),nn.Linear(1024,101))
	model.load_state_dict(torch.load('food_classifier.pt',map_location='cpu'),strict=False)
	model.eval()
	return model

# Returns a torch tensor
def get_image(image_name):
	my_transforms=transforms.Compose([
		transforms.Resize(255),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
	])
	image=Image.open(image_name)
	return my_transforms(image).unsqueeze(0)

def torch_to_np(torch_tensor):
	return torch_tensor.detach().numpy()

def process_image(image_number, model):
	print('image nr:', image_number)
	image = get_image('data/food/{:05d}.jpg'.format(image_number))
	model(image)

inner_layer_values = []
def layer_hook(m, i, o):
	inner_layer_values.append(torch_to_np(o)[0])

model = get_model()

# We extract the values from the first layer of `classifier`. dimension 1024
list(model.classifier.children())[0].register_forward_hook(layer_hook)

nr_images = 10000

# We don't care about the output of the model, instead we capture all the information
# that we need with layer_hook. There might be nicer ways to do this in pytorch but
# I don't care, this works.
for i in range(nr_images):
	process_image(i, model)

output_dir = 'data_processed'
Path(output_dir).mkdir(parents=True, exist_ok=True)
np.save(output_dir + '/torch_food.npy', np.array(inner_layer_values))
