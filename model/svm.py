"""
	Classification - SVM
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np
import cvxopt
import cvxopt.solvers
from model.classifier import Classifier

class ClassifierKernel(Classifier):
	"""
	Structure for a kernel regression classifier
	"""

	def __init__(self, kernel):
		"""
		Saves the kernel function
		"""
		self.kernel = kernel
		self.gram = None

	def project(self, data, **kwargs):
		"""
		Forecasts the output given the data
		"""
		if self.alpha is None:
			print("Train before predict")
		else:
			result = self.bias
			for z_i, x_i, y_i in zip(self.alpha,
			                         self.supportVector,
			                         self.supportVectorLabels):
				result += z_i * y_i * self.kernel(x_i, data)
			return result

	def predict(self, data, **kwargs):
		"""
		Forecast the class
		"""
		return np.sign(self.project(data))

	def train(self,trainData, trainLabels, testData = None, testLabels=None):
		"""
		Trains the model and measure on the test data
		Maximize the function sum_i_j(alpha_i alpha_j y_i y_j Kernel(x_i,x_j))
		(ie minimize opposite)
		Under the constraints 0 <= alpha <= 1
		sum_i(alpha_i y_i) = 0
 		"""
		# Computes Gram matrix
		y = trainLabels.flatten()
		self.gram = self._computeGram_(trainData)

		dim = len(trainData)

		print("\tSolve SVM Equations", end = '\r')

		"""
		min 1/2 x^T P x + q^T x
		subject to	Gx < h
					Ax = b
		"""
		# P is gram matrix
		# q is a vector of labels
		P = cvxopt.matrix(np.outer(y,y) * self.gram)
		q = cvxopt.matrix(np.ones(dim) * -1)

		# A = 1T
		A = cvxopt.matrix(y, (1,dim), 'd')
		b = cvxopt.matrix(0.0)

		# G = diag(d)
		# h = gamma*1
		G_std = np.diag(np.ones(dim) * -1)
		h_std = np.zeros(dim)

		G_slack = np.identity(dim)
		h_slack = np.ones(dim)

		G = cvxopt.matrix(np.vstack((G_std, G_slack)))
		h = cvxopt.matrix(np.hstack((h_std, h_slack)))

		# solve QP problem
		solution = cvxopt.solvers.qp(P, q, G, h, A, b)


		print("\tExtract Support Vectors", end = '\r')
		# Lagrange multipliers
		a = np.ravel(solution['x'])

		# Support vectors have non zero lagrange multipliers
		supportVector = a > 1e-8
		ind = np.arange(len(a))[supportVector]
		self.alpha = a[supportVector]
		self.supportVector = trainData[supportVector]
		self.supportVectorLabels = y[supportVector]

		# Bias
		self.bias = 0
		for n in range(len(self.alpha)):
			self.bias += self.supportVectorLabels[n]
			self.bias -= np.sum(self.alpha * self.supportVectorLabels * self.gram[ind[n],supportVector])
		self.bias /= len(self.alpha)

		self.test(trainData, trainLabels)

	def _computeGram_(self, data):
		"""
		Computes the Gram Matrix of the flatten data
		"""
		print("\tCompute Gram Matrix", end = '\r')
		dim = len(data)
		gram = np.zeros((dim, dim))
		for i in range(dim):
			for j in range(i, dim):
				gram[i,j] = self.kernel(data[i], data[j])
				gram[j,i] = gram[i,j]

		return gram
