"""
	Classification - Logistic regression
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np
from model.classifier import Classifier
from utils.regression import eveGradientDescent

class ClassifierLogistic(Classifier):
	"""
	Structure for a knn classifier
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
		return np.multiply(data,self.weight).sum() > 0

	def train(self,trainData, trainLabels, testData = None, testLabels=None):
		"""
		Trains the model and measure on the test data
		"""
		self.weight, e = eveGradientDescent(trainData, trainLabels, self.error, self.errorGrad, testData, testLabels)
		self.test(testData, testLabels)
