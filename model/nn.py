"""
	Neural Network
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np
from model.classifier import Classifier
import dataManipulation

class ClassifierNN(Classifier):
	"""
	Structure for a neural network classifier
	"""

	def __init__(self, dims, fActivation, dActivation, fCost, dCost):
		"""
		Creates a neural network respecting the different given dimensions,
		this should be a list of number, wher the first represents the number of
		trainLabels and the last, the number of outputs.
		The neural network will be fully connected
		"""
		self.layersNumber = len(dims) - 1
		self.weights = []
		self.biases = []

		self.fActivation = fActivation
		self.dActivation = dActivation
		self.fCost = fCost
		self.dCost = dCost

		for d in range(self.layersNumber):
			self.weights.append(np.random.randn(dims[d+1], dims[d]))
			self.biases.append(np.random.randn(dims[d+1], 1))

	def predict(self, data):
		"""
		Computes the result of the netword by propagation
		"""
		res = data.copy()
		for layer in range(self.layersNumber):
			weight = self.weights[layer]
			bias = self.biases[layer]
			res = self.fActivation(np.dot(weight, res) + bias)
		return np.argmax(res)

	def train(self, trainData, trainLabels, testData, testLabels,
		learningRate=0.01, regularization=0.1, batchSize=100, probabilistic=True,
		iteration=100, testTime=10):
		"""
		Computes the backpropagation of the gradient in order to reduce the
		quadratic error
		"""
		# Flattens the data
		testDataFlatten = dataManipulation.flattenImages(testData)
		trainDataFlatten = dataManipulation.flattenImages(trainData)
		trainLabelsFlatten = dataManipulation.binaryArrayFromLabel(trainLabels)

		error, pastError = 0, 0
		for ite in range(iteration):
			# Decrease the learningRate
			if ite > 1 and error > pastError :
				learningRate /= 2
			pastError = error

			# Changes order of the dataset
			if probabilistic :
				permut = np.random.permutation(len(trainDataFlatten))
				trainLabelsFlatten = trainLabelsFlatten[permut]
				trainDataFlatten = trainDataFlatten[permut]

			# Computes each image
			for batch in range(len(trainDataFlatten)//batchSize - 1):
				totalDiffWeight = [np.zeros(weight.shape) for weight in self.weights]
				totalDiffBias = [np.zeros(bias.shape) for bias in self.biases]

				# Computes the difference for each batch
				for i in range(batch*batchSize,(batch+1)*batchSize):
					diffWeight, diffBias, diffError = self._computeDiff_(trainLabelsFlatten[i], trainDataFlatten[i])
					totalDiffWeight = [totalDiffWeight[j] + diffWeight[j]
										for j in range(len(totalDiffWeight))]
					totalDiffBias = [totalDiffBias[j] + diffBias[j]
										for j in range(len(totalDiffBias))]
					error += diffError

				# Update weights and biases of each neuron
				self.weights = [self.weights[i] - learningRate*totalDiffWeight[i]
									- learningRate*regularization*self.weights[i]
									for i in range(len(totalDiffWeight))]
				self.biases = [self.biases[i] - learningRate*totalDiffBias[i]
									- learningRate*regularization*self.biases[i]
									for i in range(len(totalDiffBias))]
			print("{} / {}".format(ite+1, iteration), end = '\r')
			if ite % testTime == 0:
				self.test(testDataFlatten, testLabels)
				self.test(trainDataFlatten, trainLabels)
		self.test(testDataFlatten, testLabels)

	def _computeDiff_(self, target, input):
		"""
		Executes the forward and backward propagation for the given data
		"""
		diffWeight = [np.zeros(weight.shape) for weight in self.weights]
		diffBias = [np.zeros(bias.shape) for bias in self.biases]

		# Forward
		# layerSum contents all the result of nodes
		# layerAct = fActivation(layerSum)
		layerSum = []
		lastRes = input
		layerAct = [lastRes]
		for layer in range(self.layersNumber):
			layerRes = np.dot(self.weights[layer], lastRes) + self.biases[layer]
			lastRes = self.fActivation(layerRes)
			layerSum.append(layerRes)
			layerAct.append(lastRes)

		# Backward
		diffError = sum(self.fCost(lastRes, target))
		delta = self.dCost(lastRes, target) * self.dActivation(lastRes)
		diffBias[-1] = delta
		diffWeight[-1] = np.dot(delta, layerAct[-2].transpose())
		for layer in reversed(range(self.layersNumber-1)):
			delta = np.dot(self.weights[layer+1].transpose(), delta) *\
				self.dActivation(layerSum[layer])
			diffBias[layer] = delta
			diffWeight[layer] = np.dot(delta, layerAct[layer].transpose())

		return diffWeight, diffBias, diffError
