"""
	Classification - OneVsOne
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import dataManipulation
from model.classifier import Classifier
import numpy as np

class oneVsOne(Classifier):
	"""
	Structure in order to test a classifier with one vs all technique
	"""

	def __init__(self, model, **kwargs):
		"""
		Defines the model used for the oneVsAll classification
		"""
		self.model = model
		self.args = kwargs
		self.trainModels = []

	def predict(self, data):
		"""
		Forecasts the output given the data
		"""
		dim = len(self.trainModels)
		res = np.zeros((dim, dim))
		for i in range(len(self.trainModels)):
			for j in range(i):
				res[i, j] = self.trainModels[i][j].predict(data, **self.args)
				res[j, i] = - res[i, j]
		return np.argmax(np.sum(res,axis=1))

	def train(self, trainData, trainLabels, testData = None, testLabels=None, **kwargs):
		"""
		Trains the model and measure on the test data
		"""
		# Saves the different models and train them
		self.trainModels = []
		for i in range(10):
			modelj = []
			for j in range(i):
				print("Train i = {}, j = {}".format(i, j))
				# Creates a new model
				modelij = self.model(**self.args)
				# Selects data
				trainDatai, trainLabelsi = dataManipulation.selectLabels(trainData, trainLabels, labels=[i,j])
				testDatai, testLabelsi = dataManipulation.selectLabels(testData, testLabels, labels=[i,j])
				# Changes labels
				trainLabelsi = dataManipulation.changeLabels(trainLabelsi, label=i)
				testLabelsi = dataManipulation.changeLabels(testLabelsi, label=i)
				# Trains the new model
				modelij.train(trainDatai, trainLabelsi, testDatai, testLabelsi, **kwargs)
				modelij.test(testDatai, testLabelsi)
				modelj.append(modelij)
			self.trainModels.append(modelj)

		self.test(testData, testLabels)
