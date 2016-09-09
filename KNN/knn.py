import os
import numpy as np
from cs231n_data_utils import *

path = os.path.dirname(os.path.dirname(os.getcwd())) + '/cifar-10-batches-py/'

Xtr, Ytr, Xte, Yte = load_CIFAR10(path)

