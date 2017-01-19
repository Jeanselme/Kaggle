"""
	Test - Test the different classifiers
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import dataManipulation
from utils.distances import *
import model.knn
import model.oneVsAll
import model.logistic

# Defines the percentage of data used for training
TRAIN = 0.9

print("Extract Data and shuffles")
data = dataManipulation.extractPicturesFromCSV("data/Xtr.csv")
labels = dataManipulation.extractLabelsFromCSV("data/Ytr.csv")
length = int(len(labels)*TRAIN)

data, label = dataManipulation.shuffleDataLabel(data, labels)

# Separates between train and test sets
trainData, trainLabels = data[:length], labels[:length]
testData, testLabels = data[length:], labels[length:]

print("Test knn classifier")
knn = model.knn.ClassifierKNN(distanceKernelPoly, 5)
knn.train(trainData, trainLabels, testData, testLabels)
