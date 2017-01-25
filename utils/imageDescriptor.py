"""
	Image Descriptor
	by Vincent Jeanselme
	vincent.jeanselme@gmail.com
"""

import numpy as np

# To avoid division by 0
epsilon = 0.0000001

def normalize(array):
	"""
	Normlizes the array
	"""
	return (array - np.mean(array)) / (np.var(array))

def histogram(flatArray, downBound, upBound, numberOfInterval):
	"""
	Creates the histogram of flatArray into number of interval
	"""
	intervals = (upBound - downBound)/numberOfInterval
	res = [0] * numberOfInterval

	for data in flatArray:
		res[max(0, min(int(data - downBound// intervals), numberOfInterval -1))] += 1

	return np.array(res)

def hOg(image, numberInterval = 100, RGB = True):
	"""
	Computes the histogram of gradient
	"""
	bwimage = image.copy()
	if RGB:
		r = image[:,:,0]
		g = image[:,:,1]
		b = image[:,:,2]
		bwimage = (0.2989 * r + 0.5870 * g
		 	+ 0.1140 * b)

	gradx, grady = np.gradient(bwimage)
	alpha = np.arcsin(grady/np.sqrt(gradx**2 + grady**2 + epsilon))
	intensity = normalize(np.sqrt(gradx**2 + grady**2))

	alphaHistogram = histogram(alpha.flatten(), -np.pi/2, np.pi/2, numberInterval)
	intensityHistogram = histogram(intensity.flatten(), -1, 1, numberInterval)

	return normalize(alphaHistogram), normalize(intensityHistogram)

def hOc(image, numberInterval = 100):
	"""
	Computes the histogram of color of the image
	-> Need RGB image
	"""
	r = normalize(image[:,:,0].flatten())
	g = normalize(image[:,:,1].flatten())
	b = normalize(image[:,:,2].flatten())

	return normalize(histogram(r, -10, 10, numberInterval)), normalize(histogram(r, -10, 10, numberInterval)),\
	 	normalize(histogram(b, -10, 10, numberInterval))
