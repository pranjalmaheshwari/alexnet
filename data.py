import os
import numpy as np
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from skimage import io, transform, color
from architecture import alexnet


class ImageNetDataset(Dataset):
	# ImageNet dataset

	def __init__(self, root_dir, folder='/train/', transform=None):
		self.root_dir = root_dir
		self.labels = filter( lambda x: os.path.isdir(root_dir+x), os.listdir(root_dir))
		self.dictionary = dict(enumerate(self.labels))
		self.frames = []
		for key in self.dictionary:
			self.frames.extend( map( lambda x: [root_dir+self.dictionary[key]+folder+x, key], filter( lambda x: os.path.isfile(root_dir+self.dictionary[key]+folder+x), os.listdir(root_dir+self.dictionary[key]+folder))))
		self.transform = transform

	def __len__(self):
		return len(self.frames)

	def __getitem__(self, idx):
		img_name = self.frames[idx][0]
		image = io.imread(img_name)
		if(len(image.shape) == 2):
			image = color.gray2rgb(image)
		sample = [ image, self.frames[idx][1]]

		if self.transform:
			sample = self.transform(sample)

		return sample


class Rescale(object):
	# Rescale the image in a sample to a given size

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, labels = sample

		new_h, new_w = self.output_size, self.output_size

		new_h, new_w = int(new_h), int(new_w)

		image = transform.resize(image, (new_h, new_w))
		image = image.astype(np.float32)

		return [image, labels]


class RandomCrop(object):
	# Crop randomly the image in a sample

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):
		image, labels = sample

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h,
					  left: left + new_w]

		return [image, labels]


class ToTensor(object):
	# Convert ndarrays in sample to Tensors

	def __call__(self, sample):
		image, labels = sample

		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		image = image.transpose((2, 0, 1))
		return [torch.from_numpy(image) , labels ]


def accuracy(folder, input_path, net):
	correct = 0
	correct_top_k = 0
	total = 0
	k = 5
	transformed_dataset = ImageNetDataset(root_dir=input_path, folder=folder,
												transform=transforms.Compose([
												Rescale(224),
												ToTensor(),
											]))

	testloader = DataLoader(transformed_dataset, batch_size=80,
								shuffle=False, num_workers=4)
	
	for data in testloader:
		images, labels = data
		outputs = net(Variable(images.cuda()))
		# _, predicted = torch.max(outputs.data, 1)
		_, top_k = outputs.data.topk(k, dim=1, largest=True)
		total += labels.size(0)
		correct += (top_k[:,0] == labels.cuda()).sum()
		for j in range(k):
			correct_top_k += (top_k[:,j] == labels.cuda()).sum()

	return ((100.0 * correct) / total), ((100.0 * correct_top_k) / total), total

