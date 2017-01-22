"""
	Classification - OneVsAll
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import dataManipulation
from model.classifier import Classifier
import numpy as np

class oneVsAll(Classifier):
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
		return np.argmax([modeli.project(data, **self.args) for modeli in self.trainModels])

	def train(self, trainData, trainLabels, testData = None, testLabels=None, **kwargs):
		"""
		Trains the model and measure on the test data
		"""
		# Saves the different models and train them
		self.trainModels = []
		for i in range(max(trainLabels)-min(trainLabels)+1):
			print("Train i = {}".format(i))
			# Creates a new model
			modeli = self.model(**self.args)
			# Changes the label
			trainLabelsi = dataManipulation.changeLabels(trainLabels, label=i)
			testLabelsi = dataManipulation.changeLabels(testLabels, label=i)
			# Trains the new model
			modeli.train(trainData, trainLabelsi, testData, testLabelsi, **kwargs)
			modeli.test(testData, testLabelsi)
			self.trainModels.append(modeli)

		self.test(testData, testLabels)
