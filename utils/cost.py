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
		return (x-y)**2

	def derivateAt(self, x, y):
		return 2*(x-y)

class CrossEntropy(FunctionCost):
	"""
	Cross entropy function
	"""

	def applyTo(self, x, y):
		t = (1+y)/2
		return - np.multiply(t, np.log(x)) - np.multiply((1-t), np.log(1-x))

	def derivateAt(self, x, y):
		t = (1+y)/2
		return np.multiply(x-t, 1/np.multiply(x,1-x))

class LogisticLoss(FunctionCost):
	"""
	Logistic function
	"""

	def applyTo(self, x, y):
		return np.log(1 + np.exp(-y*x))

	def derivateAt(self, x, y):
		denum = 1 +  np.exp(y*x)
		return - y/denum

class HingeLoss(FunctionCost):
	"""
	Hinge function
	"""

	def applyTo(self, x, y):
		return max(0, 1-x*y)

	def derivateAt(self, x, y):
		if x*y < 1:
			return - x*y
		else:
			return 0
