from NaiveBayesian import *

# Load data
listPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listPosts)
trainMat = []

# Train each vector
for postInDoc in listPosts:
	trainMat.append(getBernoulliVec(myVocabList, postInDoc))

# Test 
p0Vec, p1Vec, pAbusive = trainNB0(array(trainMat), array(listClasses))
testItem = ['love', 'my', 'dalmation']
thisDoc = array(getBernoulliVec(myVocabList, testItem))
print testItem, 'classified as: ', classifyNB(thisDoc, p0Vec, p1Vec, pAbusive)

testItem = ['stupid', 'garbage']
thisDoc = array(getBernoulliVec(myVocabList, testItem))
print testItem, 'classified as: ', classifyNB(thisDoc, p0Vec, p1Vec, pAbusive)
