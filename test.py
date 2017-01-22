"""
	Test - Test the different classifiers
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import dataManipulation
from utils.distances import *
from utils.cost import *
from utils.activation import *
from utils.save import *
from utils.pca import *
import model.knn
import model.oneVsAll
import model.logistic
#import model.svm
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


print("Compute PCA")
pca = PCA(dimOutput = 2000)
trainDataPCA = pca.computePCA(dataManipulation.flattenImages(trainData))
testDataPCA = pca.applyPCA(dataManipulation.flattenImages(testData))


# KNN
print("\n Test knn classifier - Linear - 3")
knn = model.knn.ClassifierKNN(distanceKernelLinear, nearestNeighbor=3)
knn.train(trainDataPCA, trainLabels, testDataPCA, testLabels)

print("\n Test knn classifier - Poly - 3")
knn = model.knn.ClassifierKNN(distanceKernelPoly, nearestNeighbor=3)
knn.train(trainDataPCA, trainLabels, testDataPCA, testLabels)

print("\n Test knn classifier - Euclidean - 3")
knn = model.knn.ClassifierKNN(distance, 3)
knn.train(trainDataPCA, trainLabels, testDataPCA, testLabels)

print("\n Test knn classifier - RBS - 3")
knn = model.knn.ClassifierKNN(distanceKernelRBS, 3)
knn.train(trainDataPCA, trainLabels, testDataPCA, testLabels)

print("\n Test knn classifier - Radial - 3")
knn = model.knn.ClassifierKNN(distanceLaplaceRadial, 3)
knn.train(trainDataPCA, trainLabels, testDataPCA, testLabels)

print("\n Test knn classifier - Sigmoid - 3")
knn = model.knn.ClassifierKNN(distanceSigmoid, 3)
knn.train(trainDataPCA, trainLabels, testDataPCA, testLabels)


# Logistic regression
print("\n Test one vs all logistic regression")
ovaL = model.oneVsAll.oneVsAll(model.logistic.ClassifierLogistic, error=logisticLoss, errorGrad=logisticGrad)
ovaL.train(trainDataPCA, trainLabels, testDataPCA, testLabels)


# Neural Network
print("\n Test neural network")
nn = model.nn.ClassifierNN([32*32*3,1000,500,100,10], fSigmoid, dSigmoid, fQuadratic, dQuadratic)
nn.train(trainData, trainLabels, testData, testLabels)
save(nn, "NeuralNetwork-Relu-Quadratic-1000-500-100-10")
nn.test(test, save = "Yte.csv")
