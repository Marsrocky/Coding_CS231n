#coding=utf-8#
#用49000个图像作为训练集，用1000个图像作为验证集。验证集其实就是作为假的测试集来调优。

import os
import numpy as np
from cs231n_data_utils import *
from nn import *
import time

path = os.path.dirname(os.path.dirname(os.getcwd())) + '/cifar-10-batches-py/'

# Xtr: 50000 * 32 * 32 * 3, Xte: 10000 * 32 * 32 * 3
Xtr, Ytr, Xte, Yte = load_CIFAR10(path)
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32*32*3)
Xte_rows = Xte.reshape(Xte.shape[0], 32*32*3)

Xval_rows = Xtr_rows[:1000, :]
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :]
Ytr = Ytr[1000:]

start_time = time.time()

# find hyperparameters that work best
validation_accuracies = []
for k in[1, 3, 5, 10, 20, 50, 100]:
	nn = NearestNeighbor()
	nn.train(Xtr_rows, Ytr)
	Yval_predict = nn.predict(Xval_rows, k = k)
	acc = np.mean(Yval_predict == Yval)
	print 'Accuracy: %f' % (acc,)

	validation_accuracies.append((k, acc))

proper_k = 1
max_acc = 0

for i in validation_accuracies:
	if i[1] > max_acc:
		proper_k = i[0]
		max_acc = i[1]

end_time = time.time()

print 'Optimization k = %d' % proper_k
print 'Time Consumption t = ', end_time - start_time, 's'
