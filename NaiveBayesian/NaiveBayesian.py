from numpy import *

def loadDataSet():
	postingList = [
		['my', 'dog' 'has', 'flea', 'problems', 'help', 'please'],
		['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
		['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
		['stop', 'posting', 'stupid', 'worthless', 'garbage'],
		['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
		['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0, 1, 0, 1, 0, 1]
	return postingList, classVec

# Create a list with all words
def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

# Get a Bernoulli model vector for a sentence set
def getBernoulliVec(vocabList, inputSet):
	retVocabList = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			retVocabList[vocabList.index(word)] = 1
	return retVocabList

# Get a Polynomial model vector for a sentence set
def getPolyVec(vocabList, inputSet):
	retVocabList = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			retVocabList[vocabList.index(word)] += 1
	return retVocabList

# Calculate possibility of use
def trainNB0(trainMatrix, trainCategory):
	numTrainDoc = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory) / float(numTrainDoc)

	# Prevent zero possibility (as well as log calculation)
	p0Num = ones(numWords)
	p1Num = ones(numWords)
	p0Denom = 2.0; p1Denom = 2.0

	for i in range(numTrainDoc):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])

	p1Vect = log(p1Num / p1Denom)
	p0Vect = log(p0Num / p0Denom)
	return p0Vect, p1Vect, pAbusive

# Classify by comparing the possibility
def classifyNB(wordVec, p0Vec, p1Vec, pClass):
	p1 = sum(wordVec * p1Vec) + log(pClass)
	p0 = sum(wordVec * p0Vec) + log(1.0 - pClass)
	if p1 > p0:
		return 1
	else:
		return 0