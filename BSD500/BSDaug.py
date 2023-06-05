# -*- coding: utf-8 -*-
"""
Title: Context-Manager to support tracing PyTorch execution

@author: Manny Ko & Ujjawal.K.Panchal
"""
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union, Optional
import numpy as np
import skimage.exposure as skiex
import skimage.io as skio
from scipy import stats

from mk_mlutils.pipeline.augmentation_base import BaseXform as BaseXform
from mk_mlutils.pipeline import augmentation_base, augmentation


class RandomAug(BaseXform):
	""" Apply random augmentation in the 'manner' of bsd_preprocess.

	Note: bsd_preprocess have subtle statistical biases in the numpy code.
	"""
	def __init__(self, seed = 42):
		#equal probability for the following 3 flipping:
		self.flip_ops = {
			0: self.noop,
			1: self.flip_ud,
			2: self.flip_lr,
			3: self.flip_both,
		}
		self.seed = seed
		self.setNumpyRandomState_(seed)
		return

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

	def flip_both(self, images):
		return np.flip(images, axis = (1,2))

	def __call__(self, batch):
		images, labels = batch

		#fliplr = (self.ran.uniform(0.0, 1.0) > 0.5)
		#flipud = (self.ran.uniform(0.0, 1.0) > 0.5)
		op_index = self.ran.randint(0, 4) 	#uniform probability for all 4 choices
		images = self.flip_ops[op_index](images)
		labels = self.flip_ops[op_index](labels)

		#+- 50% gamma following a Gaussian/normal distribution
		#gamma = np.minimum(np.maximum(1. + self.ran.randn(), 0.5), 1.5) 	#this destroys the Gaussian
		gamma = (self.ran.randn() - 0.5) + 1 	#could just add 0.5, this is more clear - mck
		images = skiex.adjust_gamma(images, gamma)
	
		return images, labels

#reference from harmonicConvolutions.BSD500.run_BSD
def bsd_preprocess(im, tg):
	'''Data normalizations and augmentations'''
	fliplr = (np.random.rand() > 0.5)
	flipud = (np.random.rand() > 0.5)
	#+- 50% gamma following a Gaussian/normal distribution
	gamma = np.minimum(np.maximum(1. + np.random.randn(), 0.5), 1.5)
	if fliplr:
		im = np.fliplr(im)
		tg = np.fliplr(tg)
	if flipud:
		im = np.flipud(im)
		tg = np.flipud(tg)
	im = skiex.adjust_gamma(im, gamma)
	return im, tg


if __name__ == '__main__':
	crop = augmentation.RandomCrop(32)
	rot  = augmentation.Rotate(25)
	pad  = augmentation.Pad2Size(32)
	padB = augmentation.PadBoth2Size(481)

	print(crop, rot, pad, padB)

	ran = np.random.RandomState(42)

	gammas1, gammas2 = [], []
	nomals = []

	for i in range(0, 2000):
		fliplr = (ran.uniform(0.0, 1.0) > 0.5)
		flipud = (ran.uniform(0.0, 1.0) > 0.5)
		#+- 50% gamma following a Gaussian/normal distribution
		#TODO: the following code looks suspicious - mck. It relies on min/max and seems to
		#destroyed the gaussian
		n = ran.randn()
		gamma1 = np.minimum(np.maximum(1. + n, 0.5), 1.5)
		gamma2 = n - 0.5 	#(n - 0.5) + 1
		print(fliplr, flipud, gamma1)

		gammas1.append(gamma1)
		gammas2.append(gamma2)
		nomals.append(n)

	#statistic = (s^2 + k^2 (skew and kurtosis test), p-value)
	isnormal = stats.normaltest(nomals)
	gamma1_isnormal = stats.normaltest(gammas1)
	gamma2_isnormal = stats.normaltest(gammas2)
	gamma3_isnormal = stats.normaltest(np.asarray(gammas2) + 1)

	print(f"{isnormal=}")
	print(f"{gamma1_isnormal=}")
	print(f"{gamma2_isnormal=}")
	print(f"{gamma3_isnormal=}")
