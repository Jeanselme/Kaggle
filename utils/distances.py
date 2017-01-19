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
