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

	def __init__(self, dims, activationFunction, costFunction):
		"""
		Creates a neural network respecting the different given dimensions,
		this should be a list of number, wher the first represents the number of
		trainLabels and the last, the number of outputs.
		The neural network will be fully connected
		"""
		self.layersNumber = len(dims) - 1
		self.weights = []
		self.biases = []

		self.activation = activationFunction
		self.cost = costFunction

		for d in range(self.layersNumber):
			self.weights.append(np.random.randn(dims[d+1], dims[d]))
			self.biases.append(np.random.randn(dims[d+1], 1))

	def predict(self, data):
		"""
		Computes the result of the netword by propagation
		"""
		if self.weights is None:
			print("Train before predict")
		else:
			res = data.copy()
			for layer in range(self.layersNumber):
				weight = self.weights[layer]
				bias = self.biases[layer]
				res = self.activation.applyTo(np.dot(weight, res) + bias)
			return np.argmax(res)

	def test(self, testData, testLabels = None, save = None):
		"""
		Flatten data before test
		"""
		super(ClassifierNN, self).test(testData, testLabels, save)

	def train(self, trainData, trainLabels, testData, testLabels,
		learningRate=0.001, regularization=1, batchSize=100, probabilistic=True,
		iteration=1000, testTime=1,b1 = 0.9, b2 = 0.999, b3 = 0.999, epsilon = 10**(-8),
		k = 0.1, K = 10):
		"""
		Computes the backpropagation of the gradient in order to reduce the
		quadratic error
		"""
		# Flattens the data
		trainLabelsFlatten = dataManipulation.binaryArrayFromLabel(trainLabels)
		trainDataModified = trainData.copy()

		loss, oldLoss = 0, 0
		b1t, b2t = 1, 1
		d = 1

		# Moving averages
		m = [np.zeros(weight.shape) for weight in self.weights]
		v = [np.zeros(weight.shape) for weight in self.weights]

		mb = [np.zeros(bias.shape) for bias in self.biases]
		vb = [np.zeros(bias.shape) for bias in self.biases]

		for ite in range(iteration):
			print("{} / {}".format(ite+1, iteration), end = '\r')

			# Changes order of the dataset
			if probabilistic :
				trainDataModified, trainLabelsFlatten = dataManipulation.shuffleDataLabel(trainDataModified, trainLabelsFlatten)

			# Computes each image
			for batch in range(len(trainDataModified)//batchSize - 1):
				totalDiffWeight = [np.zeros(weight.shape) for weight in self.weights]
				totalDiffBias = [np.zeros(bias.shape) for bias in self.biases]

				if ite*batchSize > 1000:
					b1t = 0
					b2t = 0
				else:
					b1t *= b1
					b2t *= b2
				loss = 0

				# Computes the difference for each batch
				for i in range(batch*batchSize,(batch+1)*batchSize):
					diffWeight, diffBias, diffError = self._computeDiff_(trainLabelsFlatten[i], trainDataModified[i])
					totalDiffWeight = [totalDiffWeight[j] + diffWeight[j]/batchSize
										for j in range(len(totalDiffWeight))]
					totalDiffBias = [totalDiffBias[j] + diffBias[j]/batchSize
										for j in range(len(totalDiffBias))]
					loss += diffError

				# Moving averages for weights
				m = [b1*m[j] + (1-b1)*totalDiffWeight[j] for  j in range(len(totalDiffWeight))]
				mh = [m[j]/(1-b1t) for j in range(len(totalDiffWeight))]

				v = [b2*v[j] + (1-b2)*np.multiply(totalDiffWeight[j], totalDiffWeight[j]) for  j in range(len(totalDiffWeight))]
				vh = [v[j]/(1-b2t) for j in range(len(totalDiffWeight))]

				# Moving averages for bias
				mb = [b1*mb[j] + (1-b1)*totalDiffBias[j] for  j in range(len(totalDiffBias))]
				mbh = [mb[j]/(1-b1t) for j in range(len(totalDiffBias))]

				vb = [b2*vb[j] + (1-b2)*np.multiply(totalDiffBias[j], totalDiffBias[j]) for  j in range(len(totalDiffBias))]
				vbh = [vb[j]/(1-b2t) for j in range(len(totalDiffBias))]

				# Adaptative learning rate
				if (ite > 0):
					# In order to bound the learning rate
					if loss < oldLoss:
						delta = k + 1
						Delta = K + 1
					else:
						delta = 1/(K+1)
						Delta = 1/(k+1)
					c = min(max(delta, loss/oldLoss), Delta)
					oldLossS = oldLoss
					oldLoss = c*oldLoss
					# Computes the feedback of the error function (normalized)
					r = abs(oldLoss - oldLossS)/(min(oldLoss,oldLossS))
					# Updates the correction of learning rate
					d = b3*d + (1-b3)*r
				else:
					oldLoss = loss

				# Update weights and biases of each neuron
				self.weights = [self.weights[i] - learningRate*(np.multiply(mh[i],1/(d*np.sqrt(vh[i]) + epsilon)) + regularization*self.weights[i])
									for i in range(len(totalDiffWeight))]
				self.biases = [self.biases[i] - learningRate*(mbh[i]/(d*np.sqrt(vbh[i]) + epsilon))
									for i in range(len(totalDiffBias))]

			# Test performances
			if ite % testTime == 0:
				print(loss)
				self.test(trainData, trainLabels)
				self.test(testData, testLabels)
		self.test(testData, testLabels)

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
			lastRes = self.activation.applyTo(layerRes)
			layerSum.append(layerRes)
			layerAct.append(lastRes)

		# Backward
		diffError = sum(self.cost.applyTo(lastRes, target))
		delta = self.cost.derivateAt(lastRes, target) * self.activation.derivateAt(lastRes)
		diffBias[-1] = delta
		diffWeight[-1] = np.dot(delta, layerAct[-2].transpose())
		for layer in reversed(range(self.layersNumber-1)):
			delta = np.dot(self.weights[layer+1].transpose(), delta) *\
				self.activation.derivateAt(layerSum[layer])
			diffBias[layer] = delta
			diffWeight[layer] = np.dot(delta, layerAct[layer].transpose())

		return diffWeight, diffBias, diffError
