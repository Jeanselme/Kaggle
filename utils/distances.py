"""
	Distances
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np

def distance(image, image2):
	"""
	Computes the distance between two images of the same size
	"""
	dtrain = (image2 - image)**2
	return dtrain.sum()

def distanceKernelLinear(image, image2):
	"""
	Computes the distance between two images of the same size
	"""
	return np.multiply(image,image2).sum()

def distanceKernelPoly(image, image2, power = 5, intercept = 1):
	"""
	Computes the distance between two images of the same size
	"""
	return (np.multiply(image2,image).sum() + intercept)**power

def distanceKernelRBS(image, image2, sigma = 1):
	"""
	Computes the distance between two images of the same size
	"""
	return np.exp(-((image2-image)**2).sum()/(2*sigma**2))

def distanceLaplaceRadial(image, image2, sigma = 1):
	"""
	Computes the distance between two images of the same size
	"""
	return np.exp(-np.sqrt(((image2-image)**2).sum())/(2*sigma**2))

def distanceSigmoid(image, image2, alpha = 1, beta = 0.5):
	"""
	Computes the distance between two images of the same size
	"""
	return np.tanh(alpha + beta*np.multiply(image2,image).sum())
