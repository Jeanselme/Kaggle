"""
	Function
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

class Function:
	"""
	Object used for derivative
	"""

	def applyTo(self, x):
		"""
		Computes the value at x
		"""
		pass

	def derivateAt(self, x):
		"""
		Computes the derivative at x
		"""
		pass

class FunctionCost:
	"""
	Function used for cost computing
	"""

	def applyTo(self, x, y):
		"""
		Computes the error for prediction x instead of y
		"""
		pass

	def derivateAt(self, x, y):
		"""
		Computes the derivative at x for label y
		"""
		pass
