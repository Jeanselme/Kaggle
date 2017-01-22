"""
	Activation
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np
from utils.function import Function

class Sigmoid(Function):
	"""
	Sigmoid function
	"""

	def applyTo(self, x):
		return 1. / (1. + np.exp(-x))

	def derivateAt(self, x):
		return self.applyTo(x) * (1. - self.applyTo(x))

class SigUpd(Function):
	"""
	Sigmoid update
	"""

	def applyTo(self, x):
		return 1.7159 * np.tanh(2. / 3. * x)

	def derivateAt(self, x):
		return 1.7159 * 2. / 3. * (1. - np.tanh(2. / 3. * x)**2)

class Rectifier(Function):
	"""
	Rectifier function
	"""
	def applyTo(self, x):
		return np.log(1. + np.exp(x))

	def derivateAt(self, x):
		return 1. / (1. + np.exp(-x))

class ReLu(Function):
	"""
	ReLu function
	"""
	def applyTo(self, x):
		res = x.copy()
		res[np.where(x < 0)] = 0
		return res

	def derivateAt(self, x):
		res = x.copy()
		res[np.where(x < 0)] = 0
		res[np.where(x > 0)] = 1
		return res
