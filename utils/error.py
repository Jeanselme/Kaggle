"""
	Errors
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np

def logisticLoss(features, label, weight, bias):
	"""
	Computes the logistic loss
	"""
	return np.log(1 + np.exp(-label*(np.multiply(features,weight).sum() + bias)))

def logisticGrad(features, label, weight, bias):
	"""
	Computes the gradient of the logistic loss
	"""
	denum = 1 +  np.exp(label*(np.multiply(features,weight).sum()+bias))
	return np.multiply(features,-label/denum), -label/denum
