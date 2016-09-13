import numpy as np
from collections import Counter

class NearestNeighbor(object):
	"""KNN train and predict"""
	def __init__(self):
		pass

	def train(self, X, y):
		"""X is N x D and y is 1-d of size N"""
		self.Xtr = X
		self.ytr = y

	def predict(self, Xte, k=1):
		""" X is N x D to be predicted """
		num = Xte.shape[0]
		Ypre = np.zeros(num, dtype = self.ytr.dtype)

		# loop over all rows
		for i in xrange(num):
			# Minus every rows in X training data
			distances = np.sum(np.abs(self.Xtr-Xte[i,:]), axis=1)
			ind = distances.argsort()[:k]
			label = self.ytr[ind]
			counts = Counter(label)
			Ypre[i] = counts.most_common(1)[0][0]
			print 'Item %d is processed...' % i

		return Ypre