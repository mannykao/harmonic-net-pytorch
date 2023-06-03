'''Run BSD500'''

import argparse
import os, sys
import time
from typing import List, Tuple, Optional, Callable, Union

import numpy as np
import skimage.exposure as skiex
import skimage.io as skio
import torch

#dataset import.
from datasets.bsd500 import bsd500 as bsd500
from datasets.utils.datasetutils import getShapeMinMax
from datasets.utils.xforms import BaseXform, NullXform
from mk_mlutils.utils import torchutils
from mk_mlutils.pipeline.batch import Bagging
#from mk_mlutils.pipeline.augmentation import BaseXform

#our shared modules is one level up
sys.path.append('../')
import trainingapp


def bsd_preprocess(im:np.ndarray, tg:np.ndarray, rand:Callable=np.random.rand):
    '''Data normalizations and augmentations'''
    fliplr = (rand() > 0.5)
    flipud = (rand() > 0.5)
    gamma = np.minimum(np.maximum(1. + np.random.randn(), 0.5), 1.5)
    if fliplr:
        im = np.fliplr(im)
        tg = np.fliplr(tg)
    if flipud:
        im = np.flipud(im)
        tg = np.flipud(tg)
    im = skiex.adjust_gamma(im, gamma)
    return im, tg

class RandomFlip_Gamma(BaseXform):
	""" Null image xform 

	Args: (N/A).

	"""
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		pass

	def __call__(self, sample:np.ndarray) -> np.ndarray:
		""" getBatchAsync() yields a list - convert that to an ndarray """
		return np.asarray(sample)	#convert list to ndarray

class RandomFlip(BaseXform):
	def __init__(self, seed = 42):
		self.flip_ops = {
			0: self.noop,
			1: self.flip_ud,
			2: self.flip_lr,
		}
		self.seed = seed
		self.setNumpyRandomState_(seed)
		return
		
	def __repr__(self):
		return f"RandomFlip({self.seed=})"
	
	def setNumpyRandomState_(self, seed: Union[None, int] = None):
		s = seed if seed else self.seed
		self.ran = np.random.RandomState(s)
		return
	
	def noop(self, images):
		return images
	
	def flip_ud(self, images):
		return np.flip(images, axis = 1)
	
	def flip_lr(self, images):
		return np.flip(images, axis = 2)
	
	def __call__(self, images):
		op_index = self.ran.randint(0, 3)
		return self.flip_ops[op_index](images)

def getBSD500(kPlot=False):
	#1. get dataset.
	bsd_train, bsd_test, bsd_val = bsd500.loadBSD500(kPad=True)  #bsd500.NullXform

	if kPlot:
		#2. get instance [0].
		idx = 0
		train0 = bsd_train[idx]
		print(train0.coeffs.shape, train0.label.shape)
		if kPlot: bsd500.plotInstance(train0, title=f"BSD train[{idx}]")

		#3. plot labels.
		val0 = bsd_val[0]
		print(val0.coeffs.shape, val0.label.shape)
		if kPlot: bsd500.plotInstance(val0, title=f"BSD val[{idx}]")

		valminmax = getShapeMinMax(bsd_val)
		print(f"{valminmax=}")

	return bsd_train, bsd_test, bsd_val


if __name__ == "__main__":
	# choosing the device to run the model
	device = torchutils.onceInit(kCUDA=torch.cuda.is_available(), seed=1)

	#1. get shared args
	parser = trainingapp.shared_args(description='H-net for BSD500')
	#main(parser.parse_args())

	bsd_train, bsd_test, bsd_val = getBSD500(kPlot=False)
	print(f"{len(bsd_train)=}, {len(bsd_test)=}, {len(bsd_val)=}")