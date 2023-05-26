"""
Title: Different Instance plots for BSD500.
	
Created on Sat May 13 2023, 13:41:29.

@author: Ujjawal K. Panchal.
"""

#quintessential imports.
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

#custom libs.
from mk_mlutils.dataset import dataset_base

#dataset import.
from datasets.bsd500 import bsd500 as bsd500



if __name__ == "__main__":
	#1. get dataset.
	bsd_val = bsd500.BSD500Dataset('val')

	#2. get instance.
	instance = bsd_val[0]

	#3. plot labels.
	bsd500.plotInstance(instance)

