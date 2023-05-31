'''Run BSD500'''

import argparse
import os
#import shutil
import sys
import time
sys.path.append('../')

import numpy as np
import skimage.exposure as skiex
import skimage.io as skio

#dataset import.
from datasets.bsd500 import bsd500 as bsd500


if __name__ == "__main__":
	#1. get dataset.
	bsd_train = bsd500.BSD500Dataset('train')
	bsd_val   = bsd500.BSD500Dataset('val')

	#2. get instance [0].
	idx = 0
	train0 = bsd_train[idx]
	print(type(train0))
	bsd500.plotInstance(train0, title=f"BSD train[{idx}]")

	#3. plot labels.
	val0 = bsd_val[0]
	print(type(val0))
	bsd500.plotInstance(val0, title=f"BSD val[{idx}]")

