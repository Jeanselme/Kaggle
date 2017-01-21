"""
	Classification - Logistic regression
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np
from model.classifier import Classifier

class ClassifierLogistic(Classifier):
	"""
	Structure for a regression classifier
	"""

	def __init__(self, error, errorGrad):
		"""
		Saves the error and grad used for gradient descent
		"""
		self.error = error
		self.errorGrad = errorGrad

	def predict(self, data, **kwargs):
		"""
		Forecasts the output given the data
		"""
		return np.multiply(data,self.weight).sum() + self.bias > 0

	def train(self,trainData, trainLabels, testData = None, testLabels=None,
		maxIter = 100, learningRate = 0.1, regularization = 0.1, testTime = 100,
		b1 = 0.9, b2 = 0.999, b3 = 0.999, epsilon = 10**(-8), k = 0.1, K = 10):
		"""
		Trains the model and measure on the test data
		"""
		lossesTrain = []
		self.weight, self.bias = np.zeros(trainData[0].shape), 0

		# Moving averages
		m = np.zeros(self.weight.shape)
		v = np.zeros(self.weight.shape)
		# To compute them
		b1t = 0
		b2t = 0
		# Adaptative learning rate
		d = 1
		# Loss of the last epoch
		oldLoss = 0

		for i in range(maxIter):
			# To avoid overflow
			if i > 1000:
				b1t = 0
				b2t = 0
			else:
				b1t *= b1
				b2t *= b2
			loss = 0
			grad, gradBias = np.zeros(self.weight.shape), 0

			# Computes the full gradient and error
			for j in range(len(trainData)):
				gradLocal, gradBiasLocal = self.errorGrad(trainData[j], trainLabels[j], self.weight, self.bias)
				grad += gradLocal/len(trainData)
				gradLocal += gradBiasLocal/len(trainData)
				loss += self.error(trainData[j], trainLabels[j], self.weight, self.bias)/len(trainData)

			grad += regularization*self.weight
			gradBias += regularization*self.bias

			# Updates the moving averages
			m = b1*m + (1-b1)*grad
			mh = m / (1-b1t)

			v = b2*v + (1-b2)*np.multiply(grad,grad)
			vh = v/(1-b2t)

			# Updates the adaptative learning rate
			if (i > 0):
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

			# Updates the weight
			self.weight -= learningRate*(np.multiply(mh,1/(d*np.sqrt(vh) + epsilon)))
			self.bias -= learningRate*gradBias

			# Computes the error on the training and testing sets
			if (i % testTime == 0):
				print("Iteration : {} / {}".format(i+1, maxIter))
				print("\t-> Train Loss : {}".format(loss))
				lossesTrain.append(loss)

				if trainLabels is not None and testLabels is not None:
					loss = 0
					for j in range(len(testData)):
						loss += self.error(testData[j], testLabels[j], self.weight, self.bias)/len(testData)
					print("\t-> Test Loss : {}".format(loss))

		self.test(testData, testLabels)
