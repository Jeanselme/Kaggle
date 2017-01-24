"""
	Classification - Logistic regression
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np
import dataManipulation
from model.classifier import Classifier

class ClassifierLogistic(Classifier):
	"""
	Structure for a regression classifier
	"""

	def __init__(self, errorFunction):
		"""
		Saves the error and grad used for gradient descent
		"""
		self.error = errorFunction

	def predict(self, data, **kwargs):
		"""
		Forecasts the output given the data
		"""
		if self.weight is None:
			print("Train before predict")
		else:
			return np.sign(self.project(data))

	def project(self, data, **kwargs):
		"""
		Projects the data
		"""
		if self.weight is None:
			print("Train before predict")
		else:
			return np.multiply(data,self.weight).sum() + self.bias

	def train(self,trainData, trainLabels, testData = None, testLabels=None,
		maxIter = 1000, learningRate = 0.001, regularization = 0.01, testTime = 100,
		b1 = 0.9, b2 = 0.999, b3 = 0.999, epsilon = 10**(-8), k = 0.1, K = 10):
		"""
		Trains the model and measure on the test data
		Based on Eve algorithm for adaptative learning rate
		"""
		lossesTrain = []
		features = trainData[0].shape
		# Total number of features
		total = 1
		for d in features:
			total *= d
		self.weight, self.bias = np.random.rand(total).reshape(features), np.random.rand(1)

		# Moving averages
		m = np.zeros(self.weight.shape)
		v = np.zeros(self.weight.shape)

		mb = 0
		vb = 0

		# To compute them
		b1t = 1
		b2t = 1

		# Adaptative learning rate
		d = 1

		# Loss of the last epoch
		oldLoss = 1
		i = 0

		while i < maxIter and abs(oldLoss) > 0.1 :
			# Balance the data
			trainDataBalanced, trainLabelsBalanced = dataManipulation.balance(trainData, trainLabels, True)

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
			for j in range(len(trainDataBalanced)):
				forcast = self.project(trainDataBalanced[j])
				gradLocal = self.error.derivateAt(forcast, trainLabelsBalanced[j])/len(trainDataBalanced)
				grad += (np.multiply(gradLocal, self.weight))
				gradBias += gradLocal
				loss += self.error.applyTo(forcast, trainLabelsBalanced[j])/len(trainDataBalanced)

			# Updates the moving averages
			m = b1*m + (1-b1)*grad
			mh = m / (1-b1t)

			v = b2*v + (1-b2)*np.multiply(grad,grad)
			vh = v/(1-b2t)

			mb = b1*mb + (1-b1)*gradBias
			mbh = mb / (1-b1t)

			vb = b2*vb + (1-b2)*np.multiply(gradBias,gradBias)
			vbh = vb/(1-b2t)

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

			# Updates the weight - Regularization not taken into account in grad
			# in order to avoid a too important regularization on compaeison with grad
			self.weight -= learningRate*(np.multiply(mh,1/(d*np.sqrt(vh) + epsilon)) + regularization*self.weight/len(trainDataBalanced))
			self.bias -= learningRate*(mbh/(d*np.sqrt(vbh) + epsilon))

			i += 1

			# Computes the error on the training and testing sets
			if (i % testTime == 0):
				print("Iteration : {} / {}".format(i+1, maxIter))
				print("\t-> Train Loss : {}".format(loss))
				lossesTrain.append(loss)

				if trainLabels is not None and testLabels is not None:
					loss = 0
					for j in range(len(testData)):
						loss += self.error.applyTo(self.project(testData[j]), testLabels[j])/len(testData)
					print("\t-> Test Loss : {}".format(loss))
