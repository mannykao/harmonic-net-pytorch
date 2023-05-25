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
from urllib.request import urlopen
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from mkpyutils.testutil import time_spent
from mk_mlutils.utils import torchutils
from datasets.utils import projconfig

from datasets.mnist import mnist
from datasets.rotmnist import rotmnist


if __name__ == '__main__':
	ourdir = projconfig.getRotMNISTFolder()
	print(f"{ourdir}")

	mnist_test = mnist.MNIST(split='test')
	print(mnist_test, len(mnist_test))

	train_dataset = rotmnist.RotMNIST(split='train')
	valid_dataset = rotmnist.RotMNIST(split='valid')
	print(len(train_dataset), len(valid_dataset))
