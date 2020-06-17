import torch
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image
import numpy as np

# adapted from https://github.com/Prakhar998/Food-Classification
# Download food_classifier.pt from https://github.com/Prakhar998/Food-Classification/raw/master/food_classifier.pt

def get_model():
	model = models.densenet201(pretrained=True)
	model.classifier = nn.Sequential(nn.Linear(1920,1024),nn.LeakyReLU(),nn.Linear(1024,101))
	model.load_state_dict(torch.load('food_classifier.pt',map_location='cpu'),strict=False)
	model.eval()
	return model

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
	images_dir = '../data/food/'
	image = get_image(images_dir + "/{:05d}.jpg".format(image_number))
	return torch_to_np(model(image))[0]

inner_values = []
def layer_hook(m, i, o):
	inner_values.append(torch_to_np(o)[0])

model = get_model()
list(model.classifier.children())[0].register_forward_hook(layer_hook)

nr_images = 10000
outputs = np.array([
	process_image(i, model) for i in range(nr_images)
])

# Layer before last
np.save('preprocessed_inner_layer.npy', np.array(inner_values))

# Last layer
# np.save('preprocessed_output_layer.npy', outputs)
