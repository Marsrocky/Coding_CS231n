import os
import cPickle as pickle
import numpy as np
from scipy.misc import imread

def load_CIFAR_batch(filename):
	"""
		load a batch of data
	"""
	tf = open(filename, 'r')
	data = pickle.load(tf)
	X = data['data']
	Y = data['labels']
	X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
	Y = np.array(Y)
	return X,Y

def load_CIFAR10(ROOT):
	"""
		load whole data of cifar encoded in cpickle
	"""
	temp_x = []
	temp_y = []
	for i in range(1, 6):
		f = os.path.join(ROOT, 'data_batch_%d' % (i, ))
		X, Y = load_CIFAR_batch(f)
		temp_x.append(X)
		temp_y.append(Y)
	# Training data
	Xtr = np.concatenate(temp_x)
	Ytr = np.concatenate(temp_y)
	Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
	del X, Y, temp_x, temp_y
	return Xtr, Ytr, Xte, Yte