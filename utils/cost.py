"""
	Cost
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np

# TODO : create object with embedded derivation
def fQuadratic(x,y):
	return (x-y)**2

def dQuadratic(x,y):
	return 2*(x-y)

def fCrossEntropy(x,y):
	return - np.multiply(y, np.log(x)) + np.multiply((1-y), np.log(1-x))

def dCrossEntrop(x,y):
	return np.multiply(x-y, 1/ np.multiply(x,1-x))

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
	return - label/denum
