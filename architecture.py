import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class alexnet(nn.Module):
	# vanilla alexnet

	def __init__(self):
		super(alexnet, self).__init__()
		# conv layers
		self.conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=2)	# 3 input image channel
		self.conv2_a = nn.Conv2d(48, 128, 5, stride=1, padding=2)
		self.conv2_b = nn.Conv2d(48, 128, 5, stride=1, padding=2)
		self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
		self.conv4_a = nn.Conv2d(192, 192, 3, stride=1, padding=1)
		self.conv4_b = nn.Conv2d(192, 192, 3, stride=1, padding=1)
		self.conv5_a = nn.Conv2d(192, 128, 3, stride=1, padding=1)
		self.conv5_b = nn.Conv2d(192, 128, 3, stride=1, padding=1)
		# pool
		self.non_overlap_pool = nn.MaxPool2d(2, stride=2)
		self.overlap_pool = nn.MaxPool2d(3, stride=2)
		# dropout
		self.drop = nn.Dropout(p=0.5)
		# fully-connected: y = Wx + b
		self.fc1 = nn.Linear(256 * 6 * 6, 4096)
		self.fc2 = nn.Linear(4096, 4096)
		self.fc3 = nn.Linear(4096, 35)

	def forward(self, x):
		# layer 1
		x = F.relu(self.conv1(x))
		# layer 2
		y = F.relu(self.conv2_a(x[:,:48,:,:]))
		y = y/self.local_response_normalization(y)
		y = self.overlap_pool(y)
		z = F.relu(self.conv2_b(x[:,48:,:,:]))
		z = z/self.local_response_normalization(z)
		z = self.overlap_pool(z)
		# layer 3
		x = torch.cat((y, z), 1)
		x = x/self.local_response_normalization(x)
		x = self.overlap_pool(x)
		x = F.relu(self.conv3(x))
		# layer 4
		y = F.relu(self.conv4_a(x[:,:192,:,:]))
		z = F.relu(self.conv4_b(x[:,192:,:,:]))
		# layer 5
		y = F.relu(self.conv5_a(y))
		z = F.relu(self.conv5_b(z))
		y = self.overlap_pool(y)
		z = self.overlap_pool(z)
		# layer 6
		x = torch.cat((y,z), 1)
		x = x.view(-1, 256 * 6 * 6)
		x = F.relu(self.fc1(x))
		x = self.drop(x)
		# layer 7
		x = F.relu(self.fc2(x))
		x = self.drop(x)
		# layer 8
		x = self.fc3(x)
		return x

	def local_response_normalization(self, x):
		n = 5
		k = 2.0
		alpha = 1e-4
		beta = 0.75
		x = alpha*x*x
		x = torch.transpose(x, 1, 2)
		filters = Variable(torch.ones(x.size()[1],1,n,1).cuda())
		F.conv2d(x, filters, padding=(n/2,0), groups=x.size()[1])
		x = torch.transpose(x, 1, 2)
		x = x + k
		return x

