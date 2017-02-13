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
import model.oneVsOne
import model.logistic
import model.svm
import model.nn

# Defines the percentage of data used for training
TRAIN = 0.9

print("Extract Data")
data = dataManipulation.extractPicturesFromCSV("data/Xtr.csv")
test = dataManipulation.extractPicturesFromCSV("data/Xte.csv")
labels = dataManipulation.extractLabelsFromCSV("data/Ytr.csv")

print("Add data by rotation")
# Needs to be done before shuffling
data, labels = dataManipulation.addRotations(data, labels)
length = int(len(labels)*TRAIN)

print("Saves data")
dataManipulation.saveImagesToCsv(data, "XRotate.csv")
dataManipulation.saveLabelsToCsv(labels, "YRotate.csv")

print("Add local and global desciptor")
# Needs to be done after having all images
data = dataManipulation.describeImages(data)
test = dataManipulation.describeImages(test)

print("Shuffles data")
data, labels = dataManipulation.shuffleDataLabel(data, labels)

# Separates between train and test sets
trainData, trainLabels = data[:length], labels[:length]
testData, testLabels = data[length:], labels[length:]

print("Compute PCA")
pca = PCA(dimOutput = 2000)
trainDataPCA = pca.computePCA(trainData)
testDataPCA = pca.applyPCA(testData)

# KNN
for i in range(5,500,5):
	print("\n Test knn classifier - Linear - {}".format(i))
	knn = model.knn.ClassifierKNN(distanceKernelLinear, nearestNeighbor=i)
	knn.train(trainDataPCA, trainLabels, testDataPCA, testLabels)

	print("\n Test knn classifier - Poly - {}".format(i))
	knn = model.knn.ClassifierKNN(distanceKernelPoly, nearestNeighbor=i)
	knn.train(trainDataPCA, trainLabels, testDataPCA, testLabels)

	print("\n Test knn classifier - Euclidean - {}".format(i))
	knn = model.knn.ClassifierKNN(distance, i)
	knn.train(trainDataPCA, trainLabels, testDataPCA, testLabels)

	print("\n Test knn classifier - RBS - {}".format(i))
	knn = model.knn.ClassifierKNN(distanceKernelRBS, i)
	knn.train(trainDataPCA, trainLabels, testDataPCA, testLabels)

	print("\n Test knn classifier - Radial - {}".format(i))
	knn = model.knn.ClassifierKNN(distanceLaplaceRadial, i)
	knn.train(trainDataPCA, trainLabels, testDataPCA, testLabels)

	print("\n Test knn classifier - Sigmoid - {}".format(i))
	knn = model.knn.ClassifierKNN(distanceSigmoid, i)
	knn.train(trainDataPCA, trainLabels, testDataPCA, testLabels)

# Logistic regression
print("\n Test one vs all logistic regression")
ovaL = model.oneVsAll.oneVsAll(model.logistic.ClassifierLogistic, errorFunction=LogisticLoss())
ovaL.train(trainData, trainLabels, testData, testLabels)
ovaL.test(test, save = "Yte1.csv")

print("\n Test one vs all hinge regression")
ovaL = model.oneVsAll.oneVsAll(model.logistic.ClassifierLogistic, errorFunction=HingeLoss())
ovaL.train(trainData, trainLabels, testData, testLabels)
ovaL.test(test, save = "Yte2.csv")

print("\n Test one vs one logistic regression")
ovaL = model.oneVsOne.oneVsOne(model.logistic.ClassifierLogistic, errorFunction=LogisticLoss())
ovaL.train(trainData, trainLabels, testData, testLabels)
ovaL.test(test, save = "Yte3.csv")

print("\n Test one vs one hinge regression")
ovaL = model.oneVsOne.oneVsOne(model.logistic.ClassifierLogistic, errorFunction=HingeLoss())
ovaL.train(trainData, trainLabels, testData, testLabels)
ovaL.test(test, save = "Yte4.csv")

# Neural Network
print("\n Test neural network")
nn = model.nn.ClassifierNN([trainData[0].shape[0], 1000, 750, 500, 250 ,10], Sigmoid(), Quadratic())
nn.train(trainData, trainLabels, testData, testLabels)
save(nn, "NeuralNetwork-Relu-Quadratic-1000-750-500-250-10")
nn.test(test, save = "Yte.csv")

print("\n Test neural network")
nn = model.nn.ClassifierNN([trainData[0].shape[0], 2000, 1000, 500, 250 ,10], ReLu(), Quadratic())
nn.train(trainData, trainLabels, testData, testLabels)
save(nn, "NeuralNetwork-Relu-Quadratic-2000-1000-500-250-10")
nn.test(test, save = "Relu-Yte.csv")

print("\n Test neural network")
nn = model.nn.ClassifierNN([trainData[0].shape[0], 2000, 1000, 500, 250 ,10], Rectifier(), Quadratic())
nn.train(trainData, trainLabels, testData, testLabels)
save(nn, "NeuralNetwork-Relu-Quadratic-2000-1000-500-250-10")
nn.test(test, save = "Rectifier-Yte.csv")

print("\n Test SVM - Linear kernel - One vs all")
# -> 27%
svm = model.oneVsAll.oneVsAll(model.svm.ClassifierKernel, kernel=distanceKernelLinear)
svm.train(trainData, trainLabels, testData, testLabels)
svm.test(test, save = "YteSVMLinear.csv")

print("\n Test SVM - Polynomial kernel - One vs all")
svm = model.oneVsAll.oneVsAll(model.svm.ClassifierKernel, kernel=distanceKernelPoly)
svm.train(trainData, trainLabels, testData, testLabels)
svm.test(test, save = "YteSVM.csv")

print("\n Test SVM - Polynomial kernel - One vs one")
svm = model.oneVsOne.oneVsOne(model.svm.ClassifierKernel, kernel=distanceKernelPoly)
svm.train(trainData, trainLabels, testData, testLabels)
svm.test(test, save = "YteSVM1.csv")

print("\n Test SVM - RBS kernel - One vs one")
svm = model.oneVsOne.oneVsOne(model.svm.ClassifierKernel, kernel=distanceKernelRBS)
svm.train(trainData, trainLabels, testData, testLabels)
svm.test(test, save = "YteSVMRBS.csv")

print("\n Test SVM - Laplace kernel - One vs one")
svm = model.oneVsOne.oneVsOne(model.svm.ClassifierKernel, kernel=distanceLaplaceRadial)
svm.train(trainData, trainLabels, testData, testLabels)
svm.test(test, save = "YteSVMLaplace.csv")

print("\n Test SVM - Sigmoid kernel - One vs one")
svm = model.oneVsOne.oneVsOne(model.svm.ClassifierKernel, kernel=distanceSigmoid)
svm.train(trainData, trainLabels, testData, testLabels)
svm.test(test, save = "YteSVMSigmoid.csv")
