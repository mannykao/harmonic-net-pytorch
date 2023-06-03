"""
Title: Load the bsd500 into BSD500Dataset.
	
Created on Wed Feb 1 17:44:29 2023

@author: Manny Ko.
"""
import argparse
from pathlib import Path
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union
#import numpy as np


def shared_args(description='H-net for RotMNIST', extras:List[Tuple] =[]) -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument("--learning_rate", type=float, default=0.001, help='initially learning rate')		 	#0.076 - mck this is now the starting lr
	parser.add_argument("--n_epochs", type=int, default=20)
	parser.add_argument('--bagging', action = 'store_true', default=True, help='Bagging or DataLoader for minibatch.')
	return parser


