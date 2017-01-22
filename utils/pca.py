"""
	PCA Computation
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np
from scipy import linalg

def covariance(data):
	"""
	Computes the covariance matrix of the given data
	"""
	# Number of features
	Nf = data.shape[1]
	Cov = np.zeros((Nf, Nf))
	for i in range(Nf):
		Cov[i, i] = np.mean(data[:, i] * data[:, i])
		for k in range(Nf):
			Cov[i, k] =  np.mean(data[:, i] * data[:, k])
			Cov[k, i] = Cov[i, k]
	return Cov

class PCA:
	"""
	Principal Componenets Analysis
	"""

	def __init__(self, dimOutput = 2):
		super(PCA, self).__init__()
		self.dimOutput = dimOutput

	def computePCA(self, data):
		"""
		Computes the PCA of the given data
		(2D array : one line = one data)
		Using eigenValues decomposition

		Returns the transformed data
		"""
		dataCopy = data.copy()

		# Center data
		self.mean = np.mean(data, 0)
		self.std = np.std(data, 0)
		dataCopy -= self.mean
		dataCopy /= self.std

		# Computes the covariance matrice
		Cov = covariance(dataCopy)

		# Computes eigenvalues
		eigenValues, eigenVectors = linalg.eigh(Cov)

		# Sort them and take only the biggest one
		key = np.argsort(eigenValues)[::-1][:self.dimOutput]
		self.eigenValues, self.eigenVectors = eigenValues[key], eigenVectors[:, key]

		return self.applyPCA(data)

	def applyPCA(self, data):
		"""
		Transform the given data, applying the same transformation
		"""
		res = (data.copy() - self.mean)/self.std
		return np.dot(self.eigenVectors.T, data.T).T
