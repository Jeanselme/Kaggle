"""
	Classification - KNN
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np
from model.classifier import Classifier

class ClassifierKNN(Classifier):
	"""
	Structure for a knn classifier
	"""

	def __init__(self, distance, nearestNeighbor):
		"""
		Saves the distance to use and the number of nearest neighbors
		"""
		self.distance = distance
		self.nearestNeighbor = nearestNeighbor

	def predict(self, data, **kwargs):
		"""
		Forecasts the output given the data
		"""
		if self.trainData is None:
			print("Train before predict")
		else:
			# Computes the ditance to each data point
			distances = [self.distance(data, trainD) for trainD in self.trainData]
			labelDistance = list(zip(self.trainLabels, distances))

			# Sorts it by distance
			labelDistance.sort(key=lambda x: x[1])
			neightbor = [0] * (max(self.trainLabels) - min(self.trainLabels) + 1)

			for j in labelDistance:
				if max(neightbor) < self.nearestNeighbor :
					neightbor[j[0]] += 1
				else :
					break

			return np.argmax(neightbor) + min(self.trainLabels)

	def train(self,trainData, trainLabels, testData = None, testLabels=None):
		"""
		Trains the model and measure on the test data
		"""
		self.trainData = trainData
		self.trainLabels = trainLabels

		self.test(testData, testLabels)
