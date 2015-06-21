#############################
#
# LogRegression: Logistic Regression
# Author: Marsrock
# Email: jianfei_mars@hotmail.com
#
#############################

from numpy import *
from LogisticRegression import *
import matplotlib.pyplot as plt
import time

def loadData():
	train_x = []
	train_y = []
	fileIn = open('testData.txt')
	for line in fileIn.readlines():
		lineArr = line.strip().split()
		train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
		train_y.append(float(lineArr[2]))
	return mat(train_x), mat(train_y).transpose()

## Load data
print 'Load data'
train_x, train_y = loadData()
test_x = train_x; test_y = train_y

## Training
print 'Training'
opts = {'alpha': 0.01, 'maxIter':200, 'optimizeType':'smoothStocGradDescent'}
optimizeWeights = trainLogRegres(train_x, train_y, opts)

## Testing
print 'Testing'
accuracy = testLogRegres(optimizeWeights, test_x, test_y)

## Visualization
print 'Visualization'
print 'accuracy: %.3f%%' % (accuracy * 100)
showLogRegres(optimizeWeights, train_x, train_y)


