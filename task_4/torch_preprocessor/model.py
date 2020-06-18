import torch
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image, ImageOps
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

def process_image(image_number, model, flip=False):
	print('processing image {}, {}flipped'.format(image_number, '' if flip else 'not '))
	images_dir = '../data/food/'
	image = get_image(images_dir + "/{:05d}.jpg".format(image_number))
	if flip:
		return torch_to_np(model(torch.flip(image, [3])))[0]
	else:
		return torch_to_np(model(image))[0]

inner_values = []
def layer_hook(m, i, o):
	inner_values.append(torch_to_np(o)[0])

model = get_model()

# Record the values on the second to last layer. Sure there might be
# more elegant ways to do this in pytorch but this works.
list(model.classifier.children())[0].register_forward_hook(layer_hook)

nr_images = 10000

# We don't care about the output, the interesting values are captured
# via layer_hook.
[process_image(i, model, flip=False) for i in range(nr_images)]
np.save('preprocessed_inner_layer.npy', np.array(inner_values))

inner_values = []
[process_image(i, model, flip=True) for i in range(nr_images)]
np.save('preprocessed_inner_layer_flipped.npy', np.array(inner_values))
