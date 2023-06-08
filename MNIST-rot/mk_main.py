'''
This is the main script for training the model deep H-net model 
and evaluating its performance.

Note that choice of hyperparameters for training the model is chosen
mostly similar to the official tensorflow code of Harmonnic networks
by Worral et al, CVPR 2017, available at
https://github.com/danielewworrall/harmonicConvolutions

Author: Debjani Bhowmick, 2020.
'''

# Importing the necessary dependencies below
import argparse
import os
import random
import sys
import time
from pathlib import Path
from urllib.request import urlopen
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from mkpyutils.testutil import time_spent
from mk_mlutils.dataset import dataset_base
from mk_mlutils.utils import torchutils
from mk_mlutils.pipeline.batch import Bagging, BatchBuilder
from datasets.rotmnist import rotmnist

#our shared modules is one level up
sys.path.append('../')
from mnistmodel import DeepMNIST # for rotation equivariant CNN
from mnistmodel import RegularCNN # for regular CNN
import trainingapp


class RotMNISTDataset(rotmnist.RotMNIST):
	def __init__(self,
		name="RotMNIST", 
		split:str="train",
		device="cuda",
	):
		super().__init__(name, split=split)

		self.images = torch.from_numpy(self.images).to(device)
		self.labels = torch.from_numpy(np.asarray(self.labels, dtype=np.int64)).to(device)
		#labels = labels.type(torchutils.LongTensor)

	def __getitem__(self, idx):
		image_sample = self.images[idx]
		label_sample = self.labels[idx]
		return dataset_base.ImageDesc(image_sample, label_sample)


def download2FileAndExtract(url, folder, fileName):
	'''
	Downloads and extracts the files and folders from the 
	provided url. Note that this code has directly been taken from
	the official code of Worral et al, CVPR, 2017.

	Args:
		url (str): web link of the dataset
		folder (str): path of current folder where the zip
		will be downloaded
		filename (str): name of the zip file to be extracted
	'''

	print('Downloading rotated MNIST...')
	create_dir(folder)
	zipFileName = folder + fileName
	request = urlopen(url)
	with open(zipFileName, "wb") as f :
		f.write(request.read())
	if not zipfile.is_zipfile(zipFileName):
		print('ERROR: ' + zipFileName + ' is not a valid zip file.')
		sys.exit(1)
	print('Extracting...')
	wd = os.getcwd()
	os.chdir(folder)

	archive = zipfile.ZipFile('.'+fileName, mode='r')
	archive.extractall()
	archive.close()
	os.chdir(wd)
	print('Successfully retrieved rotated MNIST dataset.')

def settings(args):
	'''
	Contains script related to data check and initialization
	of various hyperparameters related to model training and testing. 
	Note that most parts of this function have directly been taken from
	the official code of Worral et al, CVPR, 2017.

	Returns: 
		args: argument with various hyperparameters initialized. All hyperparameters
		can be manually modified here.
		data: Rot-MNIST dataset organized into train, validation and test sets. 
	'''
	# Download MNIST if it doesn't exist
	args.dataset = 'rotated_mnist'

	if not os.path.exists(args.data_dir + '/mnist_rotation_new.zip'):
		download2FileAndExtract("https://www.dropbox.com/s/0fxwai3h84dczh0/mnist_rotation_new.zip?dl=1",
			args.data_dir, "/mnist_rotation_new.zip")
	# Load dataset
	mnist_dir = args.data_dir + '/mnist_rotation_new'
	train = np.load(mnist_dir + '/rotated_train.npz')
	valid = np.load(mnist_dir + '/rotated_valid.npz')
	test = np.load(mnist_dir + '/rotated_test.npz')
	data = {}
	data['train_x'] = train['x']
	data['train_y'] = train['y']
	data['valid_x'] = valid['x']
	data['valid_y'] = valid['y']
	data['test_x'] = test['x']
	data['test_y'] = test['y']

	
	# Other options
#   args.n_epochs = 50      #2000
	args.std_mult = 0.7
	args.delay = 12
	args.phase_preconditioner = 7.8
	args.filter_gain = 2
	args.filter_size = 5
	args.n_rings = 4
	args.n_filters = 8
	args.display_step = len(data['train_x'])/64
	args.is_classification = True
	args.dim = 28
	args.crop_shape = 0
	args.n_channels = 1
	args.n_classes = 10
	args.lr_div = 10.
	args.train_mode = True
	args.load_pretrained = False
	args.pretrained_model = './models/rotmnist_model.pth'
	args.log_path = create_dir('./logs')
	args.checkpoint_path = create_dir('./checkpoints') + '/model.ckpt'
	return args, data

def create_dir(dir_name):
	'''
	Creates the specified directory if it does not exist

	Args: 
		dir_name (str): path to the specified  directory
	'''

	if not os.path.exists(dir_name):
		os.mkdir(dir_name)
		print('Created {:s}'.format(dir_name))
	return dir_name

def main(args):
	'''
	Performs all steps from data loading, model initialization to 
	training and validation of the model.

	Args:
		args: argument variable contaning values for all the hyperparameters
	'''
	##### SETUP AND LOAD DATA #####
	args, data = settings(args)

	# choosing the device to run the model
	device = torchutils.onceInit(kCUDA=torch.cuda.is_available(), seed=1)
	#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)

	# creating train_loader and valid_loader
	#train_dataset = RotMNISTDataset(data['train_x'], data['train_y'])
	#valid_dataset = RotMNISTDataset(data['valid_x'], data['valid_y'])
	
	train_dataset = rotmnist.RotMNIST(split='train')
	test_dataset = RotMNISTDataset(split='test', device=device)
	valid_dataset = RotMNISTDataset(split='valid', device=device)
	print(f" train {len(train_dataset)}, test {len(test_dataset)}, validate {len(valid_dataset)}")

	bsize = args.batch_size
	#trainloader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, drop_last=True)
	testloader  = DataLoader(test_dataset, batch_size=bsize, drop_last=True)
	validloader = DataLoader(valid_dataset, batch_size=bsize, drop_last=True)

	if args.bagging:
		trainloader = Bagging(train_dataset, 
			batchsize=bsize, shuffle=False, drop_last=True
		)
		
	# gathering parameters for training
	lr = args.learning_rate
	max_lr = 0.076		#for CyclicLR
	print(f"Bagging {args.bagging}, {lr=}")

	model = DeepMNIST(args).to(device)
	#model = RegularCNN(args).to(device)

	if args.load_pretrained:
		model.load_state_dict(torch.load(args.pretrained_model))
	if not args.train_mode:
		args.n_epochs = 1

	params = trainingapp.TrainingParams(
		device,
		model,
		train=trainloader,
		test=testloader,
		validate=validloader,
		batchsize = args.batch_size, 			#46
		validate_batchsize = args.batch_size,	#bsize for validate and test runs
		epochs = args.n_epochs,
		lr = lr, max_lr = max_lr,
		val_best = 0.0,
		snapshot=Path(args.model_path),
	)

	# print model parameters count
	pytorch_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print('Total trainable params : ', pytorch_n_params)
	print('No. of batches : ', int(len(trainloader)/args.batch_size))
	print(f'Starting the training......{args.n_epochs}')

	# Training (10k) phase
	val_best, best_path = trainingapp.train(
		params,
		model, 
		trainloader,
		device,
		split='train'
	)
	print(" ")
	print(f"Best {val_best:.4f} {best_path=}")

	if args.best:
		snapshot = torch.load(best_path)
		model.load_state_dict(snapshot)

		# Test (50k) phase
		trainingapp.validate(
			params,
			model, 
			testloader,
			device,
			args.n_epochs,
			split='test'
		)


if __name__ == '__main__':
	#1. get shared args
	parser = trainingapp.shared_args(description='H-net for RotMNIST')
	parser.add_argument("--data_dir", default='./data', help="data directory")
	parser.add_argument("--batch_size", type=int, default=46, help="batch size")
	parser.add_argument("--model_path", type=str, default='./models/', help="snapshot path")
	args = parser.parse_args()

	main(args)
