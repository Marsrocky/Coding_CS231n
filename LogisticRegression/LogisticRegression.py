import math, time
from numpy import *
import matplotlib.pyplot as plt

# Sigmoid Function
def sigmoid(z):
	return 1/(1 + exp(-z))

# Train a logistic regression model
# Input: train_x, each row stands for a sample
#        train_y, each row is the corresponding label
#        opts, includes step and maximum number of iterations
def trainLogRegres(train_x, train_y, opts):
	startTime = time.time()

	numSamples, numFeatures = shape(train_x)
	alpha = opts['alpha']
	maxIter = opts['maxIter']
	weights = ones((numFeatures, 1))

	# optimize through gradient descent algorithm
	for k in range(maxIter):
		if opts['optimizeType'] == 'gradDescent':
			output = sigmoid(train_x * weights)
			error = train_y - output
			weights = weights + alpha * train_x.transpose() * error
		elif opts['optimizeType'] == 'stocGradDescent':
			for i in range(numSamples):
				output = sigmoid(train_x[i, :] * weights)
				error = train_y[i, 0] - output
				weights = weights + alpha * train_x[i, :].transpose() * error
		elif opts['optimizeType'] == 'smoothStocGradDescent':
			dataIndex = range(numSamples)
			for i in range(numSamples):
				alpha = 4.0 / (1.0 + k + i) + 0.01
				randIndex = int(random.uniform(0, len(dataIndex)))
				output = sigmoid(train_x[randIndex, :] * weights)
				error = train_y[randIndex, 0] - output
				weights = weights + alpha * train_x[randIndex, :].transpose() * error
				del(dataIndex[randIndex])
		else:
			raise NameError('Not support optimize method type!')

	print 'Training Complete: %fs' % (time.time() - startTime)
	return weights

# test your trained Logistic Regression model given test set
def testLogRegres(weights, test_x, test_y):
	numSamples, numFeatures = shape(test_x)
	matchCount = 0
	for i in xrange(numSamples):
		predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
		if predict == bool(test_y[i, 0]):
			matchCount += 1
	accuracy = float(matchCount) / numSamples
	return accuracy

# show the model (only availabe with 2-D)
def showLogRegres(weights, train_x, train_y):
	numSamples, numFeatures = shape(train_x)
	if numFeatures != 3:
		print "Too many dimensions"
		return 1

	# draw all samples
	for i in xrange(numSamples):
		if int(train_y[i, 0] == 0):
			plt.plot(train_x[i, 1], train_x[i, 2], 'or')
		elif int(train_x[i, 0] == 1):
			plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

	# draw the classify line
	min_x = min(train_x[:, 1])[0, 0]
	max_x = max(train_x[:, 1])[0, 0]
	weights = weights.getA()
	y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
	y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
	plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
	plt.xlabel('X1'); plt.ylabel('X2')
	plt.show()

