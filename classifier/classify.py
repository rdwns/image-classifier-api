from mxnet import nd
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model


class Classifier:

	def __init__(self):

		"""
		We're loading a pre-trained model from gluoncv-model-zoo to classify images
		"""

		self.model = get_model('cifar_resnet110_v1', classes=10, pretrained=True)

	def preprocess_image(self, image):

		"""
		We pre-process the image to be model friendly, the function performs the following transformations:
		1. Resize and crop the image to 32x32 in size,
		2. Transpose the image to num_channels*height*width,
		3. Normalize with mean and standard deviation calculated across all CIFAR10 images.
		"""

		transforms_image = transforms.Compose([transforms.Resize((32)), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
		image = transforms_image(image)
		return image


	def predict(self, image):

		#Transform the image before feeding it into the model
		image = self.preprocess_image(image)

		pred = self.model(image.expand_dims(axis=0))

		#CIFAR-10 Dataset contains only 10 classes, Only objects from the below 10 classes can be recognised by this pretrained model
		class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

		#Classify the image based on the classes defined above. The class with the most probability is returned!
		ind = nd.argmax(pred, axis=1).astype('int')
		classification = class_names[ind.asscalar()]
		probability =  nd.softmax(pred)[0][ind].asscalar()

		prediction = {
            'class': classification,
            'probability': probability
        }

		return prediction