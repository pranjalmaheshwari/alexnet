import sys
import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms, utils
import torch.optim as optim
from torch.optim import lr_scheduler
from skimage import io, transform, color
from architecture import alexnet
from data import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

inputs = sys.argv

if(len(sys.argv) < 3):
	print 'No path given'
	exit()

test = '/test/'
train = '/train/'
validation = '/validation/'

input_path = sys.argv[1]
output_file_path = sys.argv[2]

input_classes = filter( lambda x: os.path.isdir(input_path+x), os.listdir(input_path))

dictionary = dict(enumerate(input_classes))

start_time = time.time()

transformed_dataset = ImageNetDataset(root_dir=input_path,
										   transform=transforms.Compose([
											Rescale(256),
											RandomCrop(224),
											ToTensor(),
										]))

dataloader = DataLoader(transformed_dataset, batch_size=128,
						shuffle=True, num_workers=4)

print "Data loaded in time : ", (time.time()-start_time)

start_time = time.time()

net = alexnet()
net.cuda()
models_saved = 0

if(len(sys.argv) == 5):
	net.load_state_dict(torch.load(sys.argv[3]))
	models_saved = int(sys.argv[4])
	accuracy_train, accuracy_top_k_train, total_train = accuracy(train, input_path, net)
	accuracy_validation, accuracy_top_k_validation, total_validation = accuracy(validation, input_path, net)
	accuracy_test, accuracy_top_k_test, total_test = accuracy(test, input_path, net)
	print('-1 , train:, %.4f, validation:, %.4f, test:, %.4f , train:, %.4f, validation:, %.4f, test:, %.4f, total:, %d' %
			( accuracy_train, accuracy_validation, accuracy_test, 
			accuracy_top_k_train, accuracy_top_k_validation, accuracy_top_k_test,
			total_train+total_validation+total_test))

	

print "alexnet in time : ", (time.time()-start_time)
sys.stdout.flush()

it = 0
start_time = time.time()
print_time = 0.0
epochs = 100
reduce_learning_time = epochs/4

criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
weight_decay = 0.0005
momentum = 0.9
gamma = 0.1
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
scheduler = lr_scheduler.StepLR(optimizer, step_size=reduce_learning_time, gamma=gamma)

for epoch in range(1, epochs):  # loop over the dataset multiple times
	
	running_loss = 0.0
	num_iter = 0

	for i, data in enumerate(dataloader):
		# get the inputs
		inputs, labels = data

		# wrap them in Variable
		inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		num_iter += 1
		# print statistics
		running_loss += loss.data[0]

	it += num_iter

	scheduler.step()
		
	if epoch % 10 == 0:
		temp_time = time.time()
		accuracy_train, accuracy_top_k_train, total_train = accuracy(train, input_path, net)
		accuracy_validation, accuracy_top_k_validation, total_validation = accuracy(validation, input_path, net)
		accuracy_test, accuracy_top_k_test, total_test = accuracy(test, input_path, net)
		print('%d , train:, %.4f, validation:, %.4f, test:, %.4f , train:, %.4f, validation:, %.4f, test:, %.4f, total:, %d' %
				(it, accuracy_train, accuracy_validation, accuracy_test, 
				accuracy_top_k_train, accuracy_top_k_validation, accuracy_top_k_test,
				total_train+total_validation+total_test))
		torch.save(net.state_dict(), output_file_path+'_'+str(models_saved))
		sys.stdout.flush()
		models_saved += 1
		running_loss = 0.0
		print_time += time.time()-temp_time


temp_time = time.time()
accuracy_train, accuracy_top_k_train, total_train = accuracy(train, input_path, net)
accuracy_validation, accuracy_top_k_validation, total_validation = accuracy(validation, input_path, net)
accuracy_test, accuracy_top_k_test, total_test = accuracy(test, input_path, net)
print('%d , train:, %.4f, validation:, %.4f, test:, %.4f , train:, %.4f, validation:, %.4f, test:, %.4f, total:, %d' %
		(it, accuracy_train, accuracy_validation, accuracy_test, 
		accuracy_top_k_train, accuracy_top_k_validation, accuracy_top_k_test,
		total_train+total_validation+total_test))
torch.save(net.state_dict(), output_file_path+'_'+str(models_saved))
sys.stdout.flush()
models_saved += 1
print_time += time.time()-temp_time

print "Finished Training in time : ", (time.time()-start_time-print_time)
