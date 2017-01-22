"""
	Cost
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np
from utils.function import FunctionCost

class Quadratic(FunctionCost):
	"""
	Quadratic function
	"""

	def applyTo(self, x, y):
		return 1. / (1. + np.exp(-x))

	def derivateAt(self, x, y):
		return 2*(x-y)

class CrossEntropy(FunctionCost):
	"""
	Cross entropy function
	"""

	def applyTo(self, x, y):
		return - np.multiply(y, np.log(x)) + np.multiply((1-y), np.log(1-x))

	def derivateAt(self, x, y):
		return np.multiply(x-y, 1/ np.multiply(x,1-x))

class LogisticLoss(FunctionCost):
	"""
	Sigmoid function
	"""

	def applyTo(self, x, y):
		return np.log(1 + np.exp(-y*x))

	def derivateAt(self, x, y):
		denum = 1 +  np.exp(y*x)
		return - y/denum
