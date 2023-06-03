"""
Title: Load the bsd500 into BSD500Dataset.
	
Created on Wed Feb 1 17:44:29 2023

@author: Manny Ko.
"""
import argparse, time
from pathlib import Path
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union, Optional
#import numpy as np
import torch
from torch import nn

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
		lr: float = 0.001,
		lr_schedule: bool = False,
		snapshot=None,
		datasetname: Optional[str] = "RotMNIST",
		trset: Optional[str] = "test",
		seed: Optional[int] = 1,
		val_best:float = 0.0,
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
			'lr': 			lr,
			'lr_schedule':	lr_schedule,
			'snapshot': snapshot,
			'datasetname': datasetname,
			'trset': trset,
			'seed': seed,
			'val_best': val_best,
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
		pass	

	@property
	def params(self):
		return self._params

	def __repr__(self):
		return f"TrainingParams({self._params})"

	def __getitem__(self, key: str):
		return self._params.get(key, None)
#end of TrainingParams


def validate(
	params:TrainingParams,
	model:nn.Module, 
	validloader,
	model_path:str,
	device,
	epoch:int,
	split:str='validate',
) -> Tuple[float, float]:

	batch_size	= params['batchsize']
	val_best	= params['val_best']
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
			params.params['val_best'] = val_best

			# save the cuurrent model
			save_path = model_path + '/model_' + str(epoch) + '.pth'
			torch.save(model.state_dict(), save_path)

		print(f"; Val. Acc: {val_acc:.4f} ; Best: {val_best:4f} ", end="")
		tic1 = time_spent(tic0, 'validate')	

	return val_best, tic1
		

def shared_args(description='H-net for RotMNIST', extras:List[Tuple] =[]) -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument("--learning_rate", type=float, default=0.001, help='initially learning rate')		 	#0.076 - mck this is now the starting lr
	parser.add_argument("--n_epochs", type=int, default=20)
	parser.add_argument('--bagging', action = 'store_true', default=True, help='Bagging or DataLoader for minibatch.')
	return parser
