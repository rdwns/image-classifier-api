# from mxnet import nd
# from mxnet.gluon.data.vision import transforms
# from gluoncv.model_zoo import get_model

import torch
import torchvision.transforms as transforms


class Classifier:

	def __init__(self):

		"""
		We're loading a pre-trained model from torch.hub API. We automatically load the code and the pretrained weights from GitHub.-model-zoo to classify images
		"""
		self.model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)

	def preprocess_image(self, image):

		"""
		We pre-process the image to be model friendly, the function performs the following transformations:
		1. Resize and crop the image to 224x224 in size,
		2. Normalize with mean and standard deviation calculated across all CIFAR10 images.
		"""

		transforms_image = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		image = transforms_image(image)
		image = image.unsqueeze(0)

		return image


	def predict(self, image):

		#Transform the image before feeding it into the model
		image = self.preprocess_image(image)

		pred = self.model(image)
		#CIFAR-10 Dataset contains only 10 classes, Only objects from the below 10 classes can be recognised by this pretrained model
		class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
		predicted_class = torch.argmax(pred, dim=1)
		prediction = {
            'class': class_names[predicted_class],
        }

		return prediction