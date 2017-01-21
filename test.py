"""
	Test - Test the different classifiers
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import dataManipulation
from utils.distances import *
from utils.error import *
from utils.cost import *
from utils.activation import *
import model.knn
import model.oneVsAll
import model.logistic
import model.kernel
import model.nn

# Defines the percentage of data used for training
TRAIN = 0.9

print("Extract Data and shuffles")
data = dataManipulation.extractPicturesFromCSV("data/Xtr.csv")
test = dataManipulation.extractPicturesFromCSV("data/Xte.csv")
labels = dataManipulation.extractLabelsFromCSV("data/Ytr.csv")
length = int(len(labels)*TRAIN)

data, label = dataManipulation.shuffleDataLabel(data, labels)

# Separates between train and test sets
trainData, trainLabels = data[:length], labels[:length]
testData, testLabels = data[length:], labels[length:]
"""
# KNN
print("\n Test knn classifier - Linear - 5")
knn = model.knn.ClassifierKNN(distanceKernelLinear, nearestNeighbor=5)
knn.train(trainData, trainLabels, testData, testLabels)

print("\n Test knn classifier - Poly - 5")
knn = model.knn.ClassifierKNN(distanceKernelPoly, nearestNeighbor=5)
knn.train(trainData, trainLabels, testData, testLabels)

print("\n Test knn classifier - Euclidean - 3")
knn = model.knn.ClassifierKNN(distance, 3)
knn.train(trainData, trainLabels, testData, testLabels)

print("\n Test knn classifier - RBS - 5")
knn = model.knn.ClassifierKNN(distanceKernelRBS, 5)
knn.train(trainData, trainLabels, testData, testLabels)

print("\n Test knn classifier - Radial - 5")
knn = model.knn.ClassifierKNN(distanceLaplaceRadial, 5)
knn.train(trainData, trainLabels, testData, testLabels)

print("\n Test knn classifier - Sigmoid - 5")
knn = model.knn.ClassifierKNN(distanceSigmoid, 5)
knn.train(trainData, trainLabels, testData, testLabels)


# Logistic regression
print("\n Test one vs all logistic regression")
ovaL = model.oneVsAll.oneVsAll(model.logistic.ClassifierLogistic, error=logisticLoss, errorGrad=logisticGrad)
ovaL.train(trainData, trainLabels, testData, testLabels)

"""
# Neural Network
print("\n Test neural network")
nn = model.nn.ClassifierNN([32*32*3,1000,500,100,10], fSigmoid, dSigmoid, fQuadratic, dQuadratic)
nn.train(trainData, trainLabels, testData, testLabels)

# Hinge Kernel regression
print("\n Test one vs all hinge kernel regression - Linear")
ovaK = model.oneVsAll.oneVsAll(model.kernel.ClassifierKernel, kernel=distanceKernelLinear)
ovaK.train(trainData, trainLabels, testData, testLabels)

print("\n Test one vs all hinge kernel regression - Polynomial")
ovaK = model.oneVsAll.oneVsAll(model.kernel.ClassifierKernel, kernel=distanceKernelPoly)
ovaK.train(trainData, trainLabels, testData, testLabels)

print("\n Test one vs all hinge kernel regression - RBS")
ovaK = model.oneVsAll.oneVsAll(model.kernel.ClassifierKernel, kernel=distanceKernelRBS)
ovaK.train(trainData, trainLabels, testData, testLabels)

print("\n Test one vs all hinge kernel regression - Radial")
ovaK = model.oneVsAll.oneVsAll(model.kernel.ClassifierKernel, kernel=distanceLaplaceRadial)
ovaK.train(trainData, trainLabels, testData, testLabels)

print("\n Test one vs all hinge kernel regression - Sigmoid")
ovaK = model.oneVsAll.oneVsAll(model.kernel.ClassifierKernel, kernel=distanceSigmoid)
ovaK.train(trainData, trainLabels, testData, testLabels)


nn.test(test, save = "Yte.csv")
