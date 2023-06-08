"""
Title: Load the bsd500 into BSD500Dataset.
	
Created on Wed Feb 1 17:44:29 2023

@author: Manny Ko.
"""
import argparse, time
from pathlib import Path
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union, Optional
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mkpyutils.testutil import time_spent
from mk_mlutils.utils import torchutils
from mk_mlutils.pipeline.batch import Bagging


class TrainingParams():
	""" A class wrapper for a set of key-val training parameters.
		Its main purpose is to enable 'isinstance(TrainingParams)'
	Note: TrainingParams() must be able to be deepcopy(). Any object that takes a lot of memory should not be
		  placed in it.
	"""
	def __init__(self,
		device,
		model: nn.Module,
		train,
		test,
		validate,
		batchsize: int = 46,
		validate_batchsize: int = 46,	#bsize for validate and test runs
		epochs:int = 2,
		lr: float = 0.001,
		max_lr: float = 0.076,
		lr_schedule: bool = False,
		snapshot=None,
		datasetname: Optional[str] = "RotMNIST",
		trset: Optional[str] = "test",
		seed: Optional[int] = 1,
		val_best:float = 0.0,
		model_path:str = "",
		**kwargs
	):
		self._params = {
			'device':	device,
			'model': 	model,
#			'recipe':	recipe,
			'train': 	train,
			'test': 	test,
			'validate': validate,
			'batchsize': 	batchsize,
			'validate_batchsize': validate_batchsize,
			'epochs':		epochs,
			'lr': 			lr,
			'max_lr':		max_lr,
			'lr_schedule':	lr_schedule,
			'snapshot': 	snapshot,
			'datasetname': 	datasetname,
			'trset': 		trset,
			'seed': 		seed,
			'val_best': 	val_best,
		}
		self.check_params()
		self.amend(**kwargs)
		self.setup()

	def check_params(self):
		""" check to see if user passed a good set of parameters """
#		train_xform = self._params['train'].pipeline
#		test_xform  = self._params['test'].pipeline
		return True	

	def amend(self, **kwargs):
		#print(f"amend: {kwargs}")
		self._params.update(**kwargs)
		return self

	def setup(self):
		""" Callable for derived class to setup our training params using code """
		self.train_set 	= self.params['train'].dataset
		self.test_set	= self.params['test'].dataset
		self.val_set 	= self.params['validate'].dataset
		self.best_path  = ""

	@property
	def params(self):
		return self._params

	def __repr__(self):
		return f"TrainingParams({self._params})"

	def __getitem__(self, key: str):
		return self._params.get(key, None)
#end of TrainingParams


def train(
	params:TrainingParams,
	model:nn.Module, 
	trainloader,
	device,
	split:str='train',
) -> Tuple[float, Path]:

	lr 		 = params['lr']
	max_lr 	 = params['max_lr']
	n_epochs = params['epochs']
	batch_size = params['batchsize']
	validloader = params['validate']

	print(f"train({n_epochs})")

	train_dataset = params.train_set
	val_dataset   = params.val_set

	train_mode = True
	bagging = isinstance(trainloader, Bagging)

	# Optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	#optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

	#lr scheduler
	#lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
	lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
		optimizer, base_lr=lr, max_lr=max_lr, step_size_up=20, 
		mode='triangular2', cycle_momentum=False
	)
	lossfn = torch.nn.CrossEntropyLoss() # defining the loss function

	# Starting to train the model
	for epoch in range(n_epochs):
		tic1 = time.time()

		if train_mode:
			# Training phase
			model.train()

			epoch_loss = 0
			epoch_acc = 0

			correct = 0
			for idx, batch in enumerate(trainloader):
				images = batch[0]
				labels = batch[1]

				# Transfer to GPU
				if bagging:
					images = torch.from_numpy(images)
					labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
				images, labels = images.to(device), labels.to(device)
				labels = labels.type(torchutils.LongTensor)

				optimizer.zero_grad()
				logits = model(images)
				correct += (torch.argmax(logits, dim=1).type(labels.dtype)==labels).sum().item()

				#loss = lossfn(logits, labels)
				loss = F.nll_loss(F.log_softmax(logits, dim=1), labels, reduction='sum')
				epoch_loss += loss.item()
				loss.backward()

				optimizer.step()

			current_lr = lr_scheduler.get_last_lr()[0]
			if (epoch > 0) and (epoch % 100 == 0):		
				lr_scheduler.step()		#scheduler.step() should be after optimizer.step()

			epoch_acc = correct / (len(trainloader)*batch_size)
			epoch_loss /= len(train_dataset)
#			current_lr = lr_scheduler.get_lr()[0]
#			print('Epoch: ', epoch+1, '; lr: ', current_lr, '; Loss: ', epoch_loss, '; Train Acc: ', epoch_acc, end = " ")
			print(f"Epoch: {epoch+1} ; lr: {current_lr:.4f} ; Loss:  {epoch_loss:.4f} ; Train Acc: {epoch_acc:.4f}", end = " ")
			tic1 = time_spent(tic1, 'train')

		# Validation phase
		val_best = validate(
			params,
			model, 
			validloader,
			device,
			epoch=epoch+1,
			split='validate',
		)
		best_path =	params['best_path']

	return val_best, best_path

def validate(
	params:TrainingParams,
	model:nn.Module, 
	validloader,
	device,
	epoch:int,
	split:str='Val',
) -> Tuple[float]:

	batch_size	= params['batchsize']
	val_best	= params['val_best']
	model_path	= params['snapshot']
	best_path	= params['best_path']

	tic0 = time.time()
	# Validation phase

	model.eval()
	with torch.no_grad():
		val_acc = 0.0
		correct = 0
		for idx, batch in enumerate(validloader):
			images, labels = batch

			# Transfer to GPU
			images, labels = images.to(device), labels.to(device)
			labels = labels.type(torchutils.LongTensor)

			logits = model(images)
			correct += (torch.argmax(logits, dim=1).type(labels.dtype)==labels).sum().item()
		val_acc = correct / (len(validloader)*batch_size)

		if val_acc > val_best:
			val_best = val_acc

			# save the cuurrent model
			best_path = model_path/f"model_{epoch}.pth"
			torch.save(model.state_dict(), best_path)

			params.params['val_best'] = val_best
			params.params['best_path'] = best_path

		print(f"; {split}. Acc: {val_acc:.4f} ; Best: {val_best:4f} ", end="")
		tic1 = time_spent(tic0, split)	

	return val_best
		

def shared_args(description='H-net for RotMNIST', extras:List[Tuple] =[]) -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument("--learning_rate", type=float, default=0.001, help='initially learning rate')		 	#0.076 - mck this is now the starting lr
	parser.add_argument("--n_epochs", type=int, default=2)
	parser.add_argument('--bagging', action = 'store_true', default=True, help='Bagging or DataLoader for minibatch.')
	parser.add_argument('--best', type=int, default=1, help='Load best snapshot.')
	return parser


if __name__ == '__main__':
	from datasets.rotmnist import rotmnist
	from mk_mlutils.pipeline.batch import Bagging
