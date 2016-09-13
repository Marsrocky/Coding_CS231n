import os
import numpy as np
from cs231n_data_utils import *
from nn import *

path = os.path.dirname(os.path.dirname(os.getcwd())) + '/cifar-10-batches-py/'

# Xtr: 50000 * 32 * 32 * 3, Xte: 10000 * 32 * 32 * 3
Xtr, Ytr, Xte, Yte = load_CIFAR10(path)
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32*32*3)
Xte_rows = Xte.reshape(Xte.shape[0], 32*32*3)

nn = NearestNeighbor()
nn.train(Xtr_rows, Ytr)
Ypre = nn.predict(Xte_rows)

print 'Accuracy: %f' % (np.mean(Ypre == Yte))
